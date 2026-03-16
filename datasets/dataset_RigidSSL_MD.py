import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_cluster import radius_graph
from torch.nn.functional import one_hot
import torch.nn.functional as F
import mdtraj as md
from tqdm import tqdm
import glob
from utils.geometry import rot_to_quat

EPS = 1e-8

def find_trajectories(base_path):
    """Find all trajectories in the given base path pattern."""
    trajectories = []

    # Use glob to find directories matching the pattern
    dirs = glob.glob(base_path)

    for directory in dirs:
        # Look specifically for structure.pdb
        pdb_file = os.path.join(directory, "structure.pdb")

        # Find any xtc files
        xtc_files = glob.glob(os.path.join(directory, "*.xtc"))

        if os.path.exists(pdb_file) and xtc_files:
            trajectories.append({
                'dir': directory,
                'pdb': pdb_file,
                'xtc': xtc_files[0],  # Use the first XTC file
                'name': os.path.basename(os.path.dirname(directory))
            })

    return trajectories

class DatasetRigidSSLMD(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split=None,
                 index_embed_size=32, edge_embed_size=128, node_embed_size=256,
                 coordinate_scaling=0.1, use_self_conditioning=True,
                 max_neighbors=32, edge_cutoff=10.0,
                 num_bins=22, min_bin=1e-5, max_bin=20.0,
                 min_len=60, max_len=512,
                 time_interval=10, num_splits=5):
        """
        Dataset for processing MD trajectories and sampling timepoint pairs.

        Args:
            root: Root directory containing MD data
            split: Split identifier (represents a specific sampling of time frames)
            time_interval: Interval between paired timepoints (in frames)
            num_splits: Total number of splits to prepare
            Other args: Same as DatasetRigidSSLPerturb
        """
        self.split = split
        self.root = root
        # Amino acid letter to numerical encoding
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                    'N': 2, 'Y': 18, 'M': 12, "X": 20}

        # Store configuration parameters
        self.index_embed_size = index_embed_size
        self.edge_embed_size = edge_embed_size
        self.node_embed_size = node_embed_size
        self.coordinate_scaling = coordinate_scaling
        self.use_self_conditioning = use_self_conditioning
        self.max_neighbors = max_neighbors
        self.edge_cutoff = edge_cutoff
        self.num_bins = num_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.min_len = min_len
        self.max_len = max_len

        # MD-specific parameters
        self.time_interval = time_interval
        self.num_splits = num_splits

        super(DatasetRigidSSLMD, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        """Custom directory name that reflects configuration parameters."""
        parent_dir = os.path.dirname(self.root)
        config_suffix = f"_{self.node_embed_size}_{self.edge_embed_size}"
        if self.use_self_conditioning:
            config_suffix += "_sc"

        # Include MD-specific parameters in name
        name = f'processed_RigidSSL_MD_{self.time_interval}_splits{self.num_splits}{config_suffix}'
        return os.path.join(parent_dir, name, self.split)

    @property
    def raw_file_names(self):
        # We'll use the split parameter to determine which subset of trajectories to use
        return [self.split]  # Just a placeholder since raw files are MD trajectories

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        """Process MD trajectories and create paired conformations."""
        print('Processing MD trajectories...')
        os.makedirs(self.processed_dir, exist_ok=True)

        # Find all trajectories using the provided pattern
        trajectories = find_trajectories(os.path.join(self.root, "*/processed"))
        print(f"Found {len(trajectories)} MD trajectories")

        # Log file for tracking processed trajectories
        log_file = os.path.join(self.processed_dir, "processing_log.txt")
        with open(log_file, 'w') as f:
            f.write("Trajectory,Status,NumResidues,NumFrames,SampledFrames\n")

        # Process each trajectory
        data_list = []

        # Use split as random seed for consistent frame sampling across runs with same split
        # Different splits will sample different timepoints
        split_idx = int(self.split)
        np.random.seed(split_idx)

        for traj_info in tqdm(trajectories, desc="Processing trajectories"):
            pdb_file = traj_info['pdb']
            xtc_file = traj_info['xtc']
            traj_name = traj_info['name']

            try:
                # Load trajectory
                traj = md.load(xtc_file, top=pdb_file)
                topology = traj.topology

                # Get backbone atoms (N, CA, C)
                n_indices = topology.select("name N and protein")
                ca_indices = topology.select("name CA and protein")
                c_indices = topology.select("name C and protein")

                # Check if we have consistent backbone atoms
                if len(n_indices) != len(ca_indices) or len(n_indices) != len(c_indices):
                    with open(log_file, 'a') as f:
                        f.write(f"{traj_name},InconsistentBackbone,0,{traj.n_frames},0\n")
                    continue

                num_residues = len(ca_indices)

                # Apply length filtering
                if num_residues < self.min_len or num_residues > self.max_len:
                    with open(log_file, 'a') as f:
                        f.write(f"{traj_name},LengthFiltered,{num_residues},{traj.n_frames},0\n")
                    continue

                # Sample one timepoint pair for this trajectory for this specific split
                # Each split will use a different sampling of frames
                t1, t2 = self.sample_timepoint_pair(
                    traj.n_frames,
                    self.time_interval,
                    seed_offset=split_idx * 10000  # Ensure different seeds for each split
                )

                # Create a data object for this pair
                data = self.create_protein_pair_data(
                    traj, t1, t2, topology, n_indices, ca_indices, c_indices,
                    traj_id=traj_name, pair_id=split_idx
                )

                if data is not None:
                    data_list.append(data)
                    with open(log_file, 'a') as f:
                        f.write(f"{traj_name},Success,{num_residues},{traj.n_frames},t1={t1},t2={t2}\n")
                else:
                    with open(log_file, 'a') as f:
                        f.write(f"{traj_name},DataCreationFailed,{num_residues},{traj.n_frames},t1={t1},t2={t2}\n")

            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write(f"{traj_name},Error,0,0,0\n")
                print(f"Error processing {traj_name}: {e}")

        # Save the processed data
        if data_list:
            print(f"Saving {len(data_list)} processed trajectory pairs for split {split_idx}")
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            print(f"Warning: No trajectory pairs were processed successfully for split {split_idx}")
            # Create empty data to avoid errors
            empty_data = Data()
            empty_slices = {}
            torch.save((empty_data, empty_slices), self.processed_paths[0])

    def sample_timepoint_pair(self, n_frames, interval, seed_offset=0):
        """
        Sample a single pair of timepoints from a trajectory with a specific seed.

        Args:
            n_frames: Total number of frames in the trajectory
            interval: Distance between paired frames
            seed_offset: Additional seed value for randomization

        Returns:
            Tuple (t1, t2) representing a timepoint pair
        """
        if n_frames <= interval:
            # Handle edge case where trajectory is shorter than interval
            # In this case, just use first and last frame
            return 0, n_frames-1

        # Determine maximum first timepoint to allow the interval
        max_t1 = n_frames - interval - 1

        # Create trajectory-specific random seed based on trajectory properties and split
        traj_seed = hash(f"{n_frames}_{interval}_{seed_offset}") % (2**32)
        rng = np.random.RandomState(traj_seed)

        # Sample one starting timepoint
        t1 = rng.randint(0, max_t1 + 1)
        t2 = t1 + interval

        return t1, t2

    def create_protein_pair_data(self, traj, t1, t2, topology, n_indices, ca_indices, c_indices,
                                traj_id, pair_id):
        """
        Create a PyG Data object from a pair of trajectory timepoints.

        Args:
            traj: MDTraj trajectory
            t1, t2: Timepoint indices
            topology: MDTraj topology
            n_indices, ca_indices, c_indices: Indices of backbone atoms
            traj_id: Identifier for the trajectory
            pair_id: Identifier for this specific pair

        Returns:
            PyG Data object containing both conformations
        """
        try:
            # Create data object
            data = Data()

            # Get residue information
            residues = [topology.atom(i).residue for i in ca_indices]
            # Map residue names to numerical indices
            seq = []
            for residue in residues:
                aa_code = residue.name if len(residue.name) == 1 else residue.name[0]
                aa_index = self.letter_to_num.get(aa_code, 20)  # Use 20 for unknown
                seq.append(aa_index)

            # Store basic information
            data.id = f"{traj_id}_split{pair_id}_t{t1}_t{t2}"
            data.num_nodes = len(ca_indices)
            data.seq = torch.tensor(seq, dtype=torch.long)
            data.seq_idx = torch.arange(data.num_nodes, dtype=torch.long)

            # Extract coordinates for both timepoints (convert from nm to Angstrom)
            n_coords_t1 = torch.tensor(traj.xyz[t1, n_indices] * 10, dtype=torch.float32)
            ca_coords_t1 = torch.tensor(traj.xyz[t1, ca_indices] * 10, dtype=torch.float32)
            c_coords_t1 = torch.tensor(traj.xyz[t1, c_indices] * 10, dtype=torch.float32)

            n_coords_t2 = torch.tensor(traj.xyz[t2, n_indices] * 10, dtype=torch.float32)
            ca_coords_t2 = torch.tensor(traj.xyz[t2, ca_indices] * 10, dtype=torch.float32)
            c_coords_t2 = torch.tensor(traj.xyz[t2, c_indices] * 10, dtype=torch.float32)

            # Store the coordinates
            data.coords_n_t1 = n_coords_t1
            data.coords_ca_t1 = ca_coords_t1
            data.coords_c_t1 = c_coords_t1

            data.coords_n_t2 = n_coords_t2
            data.coords_ca_t2 = ca_coords_t2
            data.coords_c_t2 = c_coords_t2

            # Set self-conditioning coordinate
            data.sc_ca_t = ca_coords_t1.clone()

            # Build edge features - sparse representation (for t1 conformation)
            data.edge_index = radius_graph(
                ca_coords_t1,
                r=self.edge_cutoff,
                batch=None,
                max_num_neighbors=self.max_neighbors
            )

            # Get node features
            data.node_attr = self.get_node_features(data.seq)

            # Compute quaternions and translations
            init_quaternion_t1, init_translation_t1 = self.compute_initial_quaternions_and_translations(
                n_coords_t1, ca_coords_t1, c_coords_t1
            )

            init_quaternion_t2, init_translation_t2 = self.compute_initial_quaternions_and_translations(
                n_coords_t2, ca_coords_t2, c_coords_t2
            )

            data.init_translation_t1 = init_translation_t1
            data.init_quaternion_t1 = init_quaternion_t1

            data.init_translation_t2 = init_translation_t2
            data.init_quaternion_t2 = init_quaternion_t2

            # Store masks
            data.fixed_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.res_mask = torch.ones(data.num_nodes, dtype=torch.float32)

            # Add a default t value of 0 (will be sampled during training)
            data.t = torch.tensor([0.0], dtype=torch.float32)

            # Store rigids for compatibility
            data.rigids_t1 = torch.cat([
                data.init_quaternion_t1,
                data.init_translation_t1
            ], dim=-1)

            data.rigids_t2 = torch.cat([
                data.init_quaternion_t2,
                data.init_translation_t2
            ], dim=-1)

            return data

        except Exception as e:
            print(f"Error creating data for trajectory {traj_id}, timepoints {t1}, {t2}: {e}")
            return None

    def compute_initial_quaternions_and_translations(self, n_coords, ca_coords, c_coords):
        """
        Compute initial quaternions and translations from backbone atoms.

        Args:
            n_coords: N atom coordinates [N, 3]
            ca_coords: CA atom coordinates [N, 3]
            c_coords: C atom coordinates [N, 3]

        Returns:
            init_quaternion: [N, 4]
            init_translation: [N, 3]
        """
        num_residues = ca_coords.shape[0]
        init_quaternion = torch.zeros((num_residues, 4), device=ca_coords.device)
        init_translation = ca_coords  # [N, 3]

        # Compute local coordinate frames
        for i in range(num_residues):
            x1 = n_coords[i]
            x2 = ca_coords[i]
            x3 = c_coords[i]

            # Build local frame
            v1 = x3 - x2  # CA to C
            v2 = x1 - x2  # CA to N

            # Normalize first vector
            e1 = v1 / (torch.norm(v1) + EPS)

            # Get component of v2 orthogonal to e1
            proj = torch.dot(e1, v2) * e1
            u2 = v2 - proj

            # Normalize second vector
            e2 = u2 / (torch.norm(u2) + EPS)

            # Third vector from cross product
            e3 = torch.cross(e1, e2)

            # Create rotation matrix
            R = torch.stack([e1, e2, e3], dim=1)  # [3, 3]

            # Convert to quaternion
            quaternion = rot_to_quat(R.unsqueeze(0)).squeeze(0)
            init_quaternion[i] = quaternion

        return init_quaternion.float(), init_translation.float()

    def get_node_features(self, seq):
        """Create one-hot encoded features for amino acid types."""
        num_amino_acids = 26
        node_features = one_hot(torch.as_tensor(seq), num_classes=num_amino_acids).float()
        return node_features