import os.path as osp
import os
import numpy as np
import torch
import math
import glob
import re

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_cluster import radius_graph
from torch.nn.functional import one_hot
import torch.nn.functional as F

from utils.geometry import rot_to_quat
from tqdm import tqdm
from Bio.PDB.Polypeptide import three_to_one, is_aa

EPS = 1e-8


class DatasetRigidSSLPerturb(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split=None,
                 index_embed_size=32, edge_embed_size=128, node_embed_size=256,
                 coordinate_scaling=0.1, use_self_conditioning=True,
                 max_neighbors=32, edge_cutoff=10.0,
                 num_bins=22, min_bin=1e-5, max_bin=20.0, min_len=60, max_len=512,
                 dataset_portion="full", batch_file_dir=None):
        """
        Dataset for RigidSSL-Perturb pretraining.

        Args:
            root: Data root directory
            transform, pre_transform, pre_filter: PyG dataset parameters
            split: Which data split to use
            index_embed_size: Dimension for positional embeddings
            edge_embed_size: Dimension for edge features
            node_embed_size: Dimension for node features
            coordinate_scaling: Scaling factor for coordinates
            use_self_conditioning: Whether to use self-conditioning
            max_neighbors: Maximum number of neighbors per node
            edge_cutoff: Distance cutoff for edges
            num_bins: Number of bins for distogram
            min_bin: Minimum bin value for distogram
            max_bin: Maximum bin value for distogram
            batch_file_dir: Directory containing batch protein ID files
        """
        self.split = split
        self.root = root
        self.batch_file_dir = batch_file_dir
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
        self.dataset_portion = dataset_portion

        # File lookup cache - will be populated during processing
        self.file_lookup = {}

        super(DatasetRigidSSLPerturb, self).__init__(root, transform, pre_transform, pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    def compute_radius_of_gyration(self, coords):
        """Compute radius of gyration for a set of 3D coordinates."""
        # Center the coordinates
        center = torch.mean(coords, dim=0)
        centered_coords = coords - center

        # Compute radius of gyration
        rg = torch.sqrt(torch.mean(torch.sum(centered_coords**2, dim=1)))
        return rg.item()

    @property
    def processed_dir(self):
        """Custom directory name that reflects configuration parameters and dataset portion."""
        # Get the parent directory of self.root
        parent_dir = os.path.dirname(self.root)

        config_suffix = f"_{self.node_embed_size}_{self.edge_embed_size}"
        if self.use_self_conditioning:
            config_suffix += "_sc"

        # Add dataset portion to directory name
        name = f'processed_RigidSSL_Perturb{config_suffix}_{self.dataset_portion}'
        return osp.join(parent_dir, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'

    def get_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        """Creates embeddings for timesteps based on sinusoidal encoding."""
        if isinstance(timesteps, float) or isinstance(timesteps, int):
            timesteps = torch.tensor([timesteps])

        if len(timesteps.shape) == 0:
            # Handle scalar timestep
            timesteps = timesteps.unsqueeze(0)

        assert len(timesteps.shape) == 1
        timesteps = timesteps * max_positions
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
        )
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb

    def get_index_embedding(self, indices, embed_size, max_len=2056):
        """Creates sine/cosine positional embeddings from indices."""
        K = torch.arange(embed_size//2, device=indices.device)
        pos_embedding_sin = torch.sin(
            indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
        pos_embedding_cos = torch.cos(
            indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
        pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
        return pos_embedding

    def calc_distogram(self, pos, min_bin=None, max_bin=None, num_bins=None):
        """Calculate distance histogram for self-conditioning."""
        if min_bin is None:
            min_bin = self.min_bin
        if max_bin is None:
            max_bin = self.max_bin
        if num_bins is None:
            num_bins = self.num_bins

        dists_2d = torch.linalg.norm(
            pos[:, None, :] - pos[None, :, :], dim=-1)[..., None]
        lower = torch.linspace(
            min_bin,
            max_bin,
            num_bins,
            device=pos.device)
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        dgram = ((dists_2d > lower) * (dists_2d < upper)).float()
        return dgram

    def protein_to_graph(self, chain_df):
        """Convert protein chain DataFrame to graph Data object."""
        data = Data()

        # Extract key atom positions and amino acids
        atom_names, atom_pos, amino_types, atom_amino_id = self.parse_protein_df(chain_df)

        if atom_names is None:
            return None

        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_key_atom_pos(
            amino_types, atom_names, atom_amino_id, atom_pos
        )

        data.seq = torch.LongTensor(amino_types)
        data.coords_ca = pos_ca.float()  # Ensure float32
        data.coords_n = pos_n.float()
        data.coords_c = pos_c.float()
        data.x = atom_names
        data.pos = torch.tensor(atom_pos).float()
        data.num_nodes = len(pos_ca)

        return data

    def _three_to_one(self, residue):
        """Convert three-letter amino acid code to one-letter code."""
        try:
            return three_to_one(residue)
        except KeyError:
            return "X"

    def parse_protein_df(self, protein_df):
        """Parse protein DataFrame to extract atom information."""
        atom_names, atom_pos, residue_type, atom_amino_id = [], [], [], []
        all_residues = protein_df["residue"].unique()

        residue_num = 0
        invalid = False
        for residue in all_residues:
            residue_name = protein_df[protein_df["residue"] == residue]["resname"].iloc[0]
            if is_aa(residue_name) or residue_name == "UNK":
                residue_df = protein_df[protein_df["residue"] == residue]
                if residue_df["fullname"].str.strip().isin(["N"]).any() and residue_df["fullname"].str.strip().isin(["CA"]).any() and residue_df["fullname"].str.strip().isin(["C"]).any():
                    residue_id = self.letter_to_num[self._three_to_one(residue_name)]
                    residue_type.append(residue_id)
                    for index, row in residue_df.iterrows():
                        atom_names.append(row["fullname"].strip())
                        if [row["x"], row["y"], row["z"]] == [0., 0., 0.]:
                            invalid = True
                        atom_pos.append([row["x"], row["y"], row["z"]])
                        atom_amino_id.append(residue_num)

                    residue_num += 1

        if invalid:
            return None, None, None, None

        return atom_names, np.array(atom_pos), residue_type, np.array(atom_amino_id)

    def get_key_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        """Extract positions of key atoms from protein structure."""
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, 'N')
        mask_ca = np.char.equal(atom_names, 'CA')
        mask_c = np.char.equal(atom_names, 'C')
        mask_cb = np.char.equal(atom_names, 'CB')
        mask_g = np.char.equal(atom_names, 'CG') | np.char.equal(atom_names, 'SG') | np.char.equal(atom_names, 'OG') | np.char.equal(atom_names, 'CG1') | np.char.equal(atom_names, 'OG1')
        mask_d = np.char.equal(atom_names, 'CD') | np.char.equal(atom_names, 'SD') | np.char.equal(atom_names, 'CD1') | np.char.equal(atom_names, 'OD1') | np.char.equal(atom_names, 'ND1')
        mask_e = np.char.equal(atom_names, 'CE') | np.char.equal(atom_names, 'NE') | np.char.equal(atom_names, 'OE1')
        mask_z = np.char.equal(atom_names, 'CZ') | np.char.equal(atom_names, 'NZ')
        mask_h = np.char.equal(atom_names, 'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h

    def read_batch_file(self):
        """Read protein IDs from the batch file corresponding to the current split."""
        if self.batch_file_dir is None:
            return None

        batch_file = os.path.join(self.batch_file_dir, f"batch_{self.split}_proteins.txt")
        if not os.path.exists(batch_file):
            print(f"Warning: Batch file {batch_file} not found.")
            return None

        protein_ids = []
        with open(batch_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Extract just the base protein ID by removing chain ID and processing duplicated part
                # Format is like: AF-Q2LQM3-F1-model_v4_AF-Q2LQM3-F1-model_v4.pdb_A,80
                parts = line.split(',')
                if len(parts) >= 1:
                    full_id = parts[0]
                    # Remove chain ID (everything after last underscore)
                    if '_' in full_id:
                        base_id = full_id.rsplit('_', 1)[0]
                    else:
                        base_id = full_id

                    # Handle duplicated model IDs - detect if "AF-" appears twice
                    if base_id.count("AF-") > 1:
                        # Extract just the first part for AlphaFold models
                        match = re.search(r'(AF-[A-Za-z0-9]+-F\d+-model_v\d+)', base_id)
                        if match:
                            base_id = match.group(1)

                    # Remove .pdb extension if present
                    if base_id.endswith('.pdb'):
                        base_id = base_id[:-4]

                    protein_ids.append(base_id)

        return protein_ids

    def build_file_index(self):
        """Build an index of PDB files to avoid expensive filesystem searches."""
        print("Building PDB file index - this will speed up processing...")

        # Strategy 1: Direct search in root directory
        pdb_files = glob.glob(os.path.join(self.root, "*.pdb"))
        for pdb_file in pdb_files:
            basename = os.path.basename(pdb_file)
            protein_id = basename.replace(".pdb", "")
            self.file_lookup[protein_id] = pdb_file

        print(f"Found {len(self.file_lookup)} PDB files in root directory")

    def find_pdb_file(self, protein_id):
        """Find the PDB file path for a given protein ID using the file index."""
        # Direct path construction - this is the main method given the file name format
        direct_path = os.path.join(self.root, f"{protein_id}.pdb")
        if os.path.exists(direct_path):
            return direct_path

        # Check if it's in our lookup dictionary (unlikely to be needed given the clear naming)
        if protein_id in self.file_lookup:
            return self.file_lookup[protein_id]

        # Not found
        return None

    def process(self):
        """Process PDB files into graph Data objects using pre-filtered batch files."""
        print('Beginning Processing...')
        print("Processing from", self.root)

        # Set up logging
        filter_stats = {
            "total": 0,
            "valid": 0,
            "not_found": 0,
            "invalid": 0,
        }

        log_file = os.path.join(self.processed_dir, "process_log.txt")
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write("Protein_ID,Status,Length\n")

        # Read protein IDs from batch file
        protein_ids = self.read_batch_file()
        if protein_ids is None or len(protein_ids) == 0:
            print("No protein IDs found in batch file. Exiting.")
            # Create empty data to avoid errors
            empty_data = Data()
            empty_slices = {}
            torch.save((empty_data, empty_slices), self.processed_paths[0])
            return

        print(f"Found {len(protein_ids)} protein IDs in batch file.")

        # Build file index to speed up lookups if needed
        # (Though with the clear naming convention, direct path construction is likely sufficient)
        self.build_file_index()

        # Load and process the proteins
        filtered_data_list = []

        for protein_id in tqdm(protein_ids, desc="Processing proteins"):
            filter_stats["total"] += 1

            # Find the PDB file
            pdb_file = self.find_pdb_file(protein_id)
            if pdb_file is None:
                filter_stats["not_found"] += 1
                with open(log_file, 'a') as f:
                    f.write(f"{protein_id},NotFound,0\n")
                continue

            try:
                # Load the protein structure
                import atom3d.util.formats as fo
                protein = fo.read_any(pdb_file)
                protein_df = fo.bp_to_df(protein)

                # Process the protein (assume single chain)
                data = self.protein_to_graph(protein_df)

                if data is None:
                    filter_stats["invalid"] += 1
                    with open(log_file, 'a') as f:
                        f.write(f"{protein_id},Invalid,0\n")
                    continue

                # Add protein ID
                data.id = protein_id

                # Continue with regular processing
                data.node_attr = self.get_node_features(data.seq)  # [N, 26]

                # Add sequence indices for embedding
                num_residues = data.node_attr.shape[0]
                data.seq_idx = torch.arange(num_residues, dtype=torch.long)
                data.chain_idx = torch.zeros(num_residues, dtype=torch.long)  # Single chain

                # Build edge features - sparse representation
                data.edge_index = radius_graph(
                    data.coords_ca,
                    r=self.edge_cutoff,
                    batch=None,
                    max_num_neighbors=self.max_neighbors
                )

                # For direct prediction approach, store the original coordinates
                # These will be replaced by interpolated coordinates during training
                data.sc_ca_t = data.coords_ca.clone()
                data.orig_ca = data.coords_ca.clone()  # Keep original for reference

                # Compute quaternions and translations
                init_quaternion, init_translation = self.compute_initial_quaternions_and_translations(data)
                data.init_quaternion = init_quaternion
                data.init_translation = init_translation

                # Store masks
                data.fixed_mask = torch.zeros(num_residues, dtype=torch.bool)
                data.res_mask = torch.ones(num_residues, dtype=torch.float32)

                # Add a default t value of 0 (will be sampled during training)
                data.t = torch.tensor([0.0], dtype=torch.float32)

                # Create rigids_t (quaternion + translation)
                data.rigids_t = torch.cat([
                    data.init_quaternion,
                    data.init_translation
                ], dim=-1)

                # Make sure all tensors are float32 by default
                for key, value in data:
                    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                        data[key] = value.float()

                filtered_data_list.append(data)
                filter_stats["valid"] += 1

                with open(log_file, 'a') as f:
                    f.write(f"{protein_id},Valid,{num_residues}\n")

            except Exception as e:
                filter_stats["invalid"] += 1
                with open(log_file, 'a') as f:
                    f.write(f"{protein_id},Error: {str(e)},0\n")

        # Print statistics
        print("\nProcessing Statistics:")
        print(f"Total proteins from batch file: {filter_stats['total']}")
        print(f"PDB files not found: {filter_stats['not_found']}")
        print(f"Invalid proteins: {filter_stats['invalid']}")
        print(f"Valid proteins processed: {filter_stats['valid']}")
        print(f"Detailed processing log saved to: {log_file}")

        # Save the data
        if filtered_data_list:
            data, slices = self.collate(filtered_data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            print("Warning: No proteins processed successfully!")
            # Create empty data to avoid errors
            empty_data = Data()
            empty_slices = {}
            torch.save((empty_data, empty_slices), self.processed_paths[0])

    def get_node_features(self, seq):
        """Create one-hot encoded features for amino acid types."""
        num_amino_acids = 26
        node_features = one_hot(torch.as_tensor(seq), num_classes=num_amino_acids).float()
        return node_features  # [N, 26]

    def compute_initial_quaternions_and_translations(self, protein):
        """Compute initial quaternions and translations from backbone atoms."""
        N_coords = protein.coords_n  # [N, 3]
        CA_coords = protein.coords_ca  # [N, 3]
        C_coords = protein.coords_c  # [N, 3]

        num_residues = CA_coords.shape[0]
        init_quaternion = torch.zeros((num_residues, 4), device=CA_coords.device)
        init_translation = CA_coords  # [N, 3]

        for i in range(num_residues):
            x1 = N_coords[i]
            x2 = CA_coords[i]
            x3 = C_coords[i]
            v1 = x3 - x2  #CA to C
            v2 = x1 - x2  #CA to N
            e1 = v1 / (torch.norm(v1) + EPS)
            proj = torch.dot(e1, v2) * e1  # projection onto e1
            u2 = v2 - proj
            e2 = u2 / (torch.norm(u2) + EPS)
            e3 = torch.cross(e1, e2)
            R = torch.stack([e1, e2, e3], dim=1)  # [3, 3]
            quaternion = rot_to_quat(R.unsqueeze(0)).squeeze(0)
            init_quaternion[i] = quaternion

        return init_quaternion.float(), init_translation.float()  # [N, 4], [N, 3]

    def prepare_inputs(self, data, t_value=None):
        """
        Prepare inputs for the VelocityNetwork.

        Args:
            data: PyG Data object
            t_value: Optional timestep value (0.0-1.0)

        Returns:
            Dictionary with formatted inputs for VelocityNetwork
        """
        # Use provided t_value or default to data.t
        t = t_value if t_value is not None else data.t
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)

        # Prepare input dictionary with all required fields
        input_feats = {
            'seq_idx': data.seq_idx,
            't': t,
            'fixed_mask': data.fixed_mask,
            'sc_ca_t': data.sc_ca_t,  # This will be updated dynamically during training
            'res_mask': data.res_mask,
            'rigids_t': data.rigids_t
        }

        return input_feats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--min_len", type=int, default=60)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--dataset_portion", type=str, default="full",
                       choices=["full", "10percent", "1percent"])
    parser.add_argument("--input_data_dir", type=str,
                       default="")
    parser.add_argument("--batch_file_dir", type=str,
                       default="")
    args = parser.parse_args()

    print(f'Processing split {args.split} with batch file from {args.batch_file_dir}')
    dataset = DatasetRigidSSLPerturb(
        root=args.input_data_dir,
        split=args.split,
        min_len=args.min_len,
        max_len=args.max_len,
        dataset_portion=args.dataset_portion,
        batch_file_dir=args.batch_file_dir
    )
    print(f"Processed dataset with {len(dataset)} proteins")