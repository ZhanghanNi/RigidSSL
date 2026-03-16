import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from config import args
from utils.rotation import extract_rotation_matrix
from model.velocity_network import VelocityNetwork
from utils.geometry import quat_to_rot, rot_to_quat, SLERP, LERP, SLERP_derivative_edge
from datasets import DatasetRigidSSLMD


def align_to_shared_global_frame(x1, r1, x2, r2):
    x_combined = torch.cat([x1, x2], dim=0)
    shared_center = torch.mean(x_combined, dim=0, keepdim=True)
    x1_centered = x1 - shared_center
    x2_centered = x2 - shared_center

    x_combined_centered = torch.cat([x1_centered, x2_centered], dim=0)
    shared_inertia_frame, _ = extract_rotation_matrix(x_combined_centered)

    r1_aligned = torch.matmul(r1, shared_inertia_frame)
    r2_aligned = torch.matmul(r2, shared_inertia_frame)

    return x1_centered, r1_aligned, x2_centered, r2_aligned


def align_global_frame(x, r):
    center = torch.mean(x, dim=0, keepdim=True)
    x_aligned = x - center

    inertia_frame, _ = extract_rotation_matrix(x_aligned)
    r_aligned = torch.matmul(r, inertia_frame)

    return x_aligned, r_aligned


def create_model_config():
    class ModelConfig:
        def __init__(self):
            self.node_embed_size = args.ipa_s_dim
            self.edge_embed_size = args.ipa_z_dim
            self.dropout = 0.0

            self.embed = type('EmbedConfig', (), {
                'index_embed_size': args.t_emb_dim,
                'aatype_embed_size': 64,
                'embed_self_conditioning': True,
                'num_bins': 22,
                'min_bin': 1e-5,
                'max_bin': 20.0,
            })
            self.ipa = type('IPAConfig', (), {
                'c_s': args.ipa_s_dim,
                'c_z': args.ipa_z_dim,
                'c_hidden': 256,
                'c_skip': 64,
                'no_heads': args.ipa_num_heads,
                'no_qk_points': 8,
                'no_v_points': 12,
                'seq_tfmr_num_heads': 4,
                'seq_tfmr_num_layers': 2,
                'num_blocks': args.ipa_num_blocks,
                'coordinate_scaling': 0.1,
                            'velocity_head_hidden': getattr(args, 'velocity_head_hidden', args.ipa_s_dim),
                'velocity_head_layers': getattr(args, 'velocity_head_layers', 2),
            })

    return ModelConfig()


def model_setup():
    model_conf = create_model_config()
    model = VelocityNetwork(model_conf)
    return model


def save_model(model, optimizer, lr_scheduler, epoch=None, save_best=False):
    if not args.output_model_dir == "":
        base_dir = args.output_model_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        time_interval_suffix = f"_interval{args.time_interval}" if hasattr(args, 'time_interval') else ""

        if save_best:
            global optimal_loss
            print(f"save model with loss: {optimal_loss:.5f}")
            output_model_path = os.path.join(base_dir, f"model{time_interval_suffix}.pth")
        else:
            print(f"save model in epoch {epoch}")
            output_model_path = os.path.join(base_dir, f"model_{epoch}{time_interval_suffix}.pth")
        saved_model_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if lr_scheduler is not None:
            saved_model_dict["scheduler"] = lr_scheduler.state_dict()

        torch.save(saved_model_dict, output_model_path)

    return


def prepare_input_features(protein, x_t, q_t, t, device):
    seq_idx = protein.seq_idx.unsqueeze(0).to(device)
    fixed_mask = protein.fixed_mask.unsqueeze(0).to(device)
    res_mask = protein.res_mask.unsqueeze(0).to(device)

    x_t = x_t.unsqueeze(0).to(device)
    q_t = q_t.unsqueeze(0).to(device)

    sc_ca_t = x_t

    rigids_t = torch.cat([q_t, x_t], dim=-1)

    input_feats = {
        'seq_idx': seq_idx,
        't': torch.tensor([t], device=device),
        'fixed_mask': fixed_mask,
        'sc_ca_t': sc_ca_t,
        'res_mask': res_mask,
        'rigids_t': rigids_t
    }

    return input_feats

def run_model_on_protein(model, protein, x0, x1, q0, q1, t, device, direction='forward'):
    """Run model with specified direction (forward or backward)."""
    xt = LERP(x0, x1, t)
    qt = SLERP(q0, q1, t)
    xt = xt.float()
    qt = qt.float()

    input_feats = prepare_input_features(protein, xt, qt, t, device)

    node_embed, v_trans_pred, v_rot_pred = model(input_feats, direction=direction)

    if len(node_embed.shape) == 3 and node_embed.shape[0] == 1:
        node_embed = node_embed.squeeze(0)
    if len(v_trans_pred.shape) == 3 and v_trans_pred.shape[0] == 1:
        v_trans_pred = v_trans_pred.squeeze(0)
    if len(v_rot_pred.shape) == 3 and v_rot_pred.shape[0] == 1:
        v_rot_pred = v_rot_pred.squeeze(0)

    return node_embed, v_trans_pred, v_rot_pred

def train(model, device, loader, optimizer, args):
    model.train()
    accum_loss = 0
    accum_loss_trans_fwd = 0
    accum_loss_rot_fwd = 0
    accum_loss_trans_bwd = 0
    accum_loss_rot_bwd = 0

    L = tqdm(loader)
    for step, batch in enumerate(L):
        for i in range(batch.num_graphs):
            protein_idx = i
            start_idx = batch.ptr[protein_idx].item()
            end_idx = batch.ptr[protein_idx + 1].item()
            protein = type('', (), {})()
            protein.seq_idx = batch.seq_idx[start_idx:end_idx].to(device)
            protein.fixed_mask = batch.fixed_mask[start_idx:end_idx].to(device)
            protein.sc_ca_t = batch.sc_ca_t[start_idx:end_idx].to(device)
            protein.res_mask = batch.res_mask[start_idx:end_idx].to(device)

            x1 = batch.init_translation_t1[start_idx:end_idx].to(device)
            q1 = batch.init_quaternion_t1[start_idx:end_idx].to(device)
            q1 = q1 / torch.norm(q1, p=2, dim=-1, keepdim=True)
            mask1 = q1[..., 0] < 0
            q1[mask1] *= -1

            x2 = batch.init_translation_t2[start_idx:end_idx].to(device)
            q2 = batch.init_quaternion_t2[start_idx:end_idx].to(device)
            q2 = q2 / torch.norm(q2, p=2, dim=-1, keepdim=True)
            mask2 = q2[..., 0] < 0
            q2[mask2] *= -1

            r1 = quat_to_rot(q1)
            r2 = quat_to_rot(q2)

            x1_aligned, r1_aligned, x2_aligned, r2_aligned = align_to_shared_global_frame(x1, r1, x2, r2)

            q1_aligned = rot_to_quat(r1_aligned)
            q2_aligned = rot_to_quat(r2_aligned)

            x1, q1 = x1_aligned, q1_aligned
            x2, q2 = x2_aligned, q2_aligned

            sample_t_values = torch.rand(args.train_number)

            for j in range(args.train_number):
                t_val = sample_t_values[j].item()

                v_trans_target_fwd = x2 - x1

                t_tensor = torch.tensor(t_val, device=device).unsqueeze(0).expand(q1.shape[0], 1)
                v_rot_target_fwd = SLERP_derivative_edge(q1, q2, t_tensor)

                _, v_trans_pred_fwd, v_rot_pred_fwd = run_model_on_protein(
                    model, protein, x1, x2, q1, q2, t_val, device, direction='forward'
                )

                trans_std_fwd = v_trans_target_fwd.std() + 1e-8
                rot_std_fwd = v_rot_target_fwd.std() + 1e-8

                v_trans_target_fwd_norm = v_trans_target_fwd / trans_std_fwd
                v_rot_target_fwd_norm = v_rot_target_fwd / rot_std_fwd

                v_trans_pred_fwd_norm = v_trans_pred_fwd / trans_std_fwd
                v_rot_pred_fwd_norm = v_rot_pred_fwd / rot_std_fwd

                loss_t_fwd = torch.mean((v_trans_target_fwd_norm - v_trans_pred_fwd_norm) ** 2, dim=-1)
                loss_r_fwd = torch.mean((v_rot_target_fwd_norm - v_rot_pred_fwd_norm) ** 2, dim=-1)

                trans_weight = getattr(args, 'trans_loss_weight', 10)
                rot_weight = getattr(args, 'rot_loss_weight', 1)
                loss_fwd = 2 * trans_weight * loss_t_fwd + rot_weight * loss_r_fwd

                v_trans_target_bwd = x1 - x2
                v_rot_target_bwd = SLERP_derivative_edge(q2, q1, t_tensor)

                _, v_trans_pred_bwd, v_rot_pred_bwd = run_model_on_protein(
                    model, protein, x2, x1, q2, q1, t_val, device, direction='backward'
                )

                trans_std_bwd = v_trans_target_bwd.std() + 1e-8
                rot_std_bwd = v_rot_target_bwd.std() + 1e-8

                v_trans_target_bwd_norm = v_trans_target_bwd / trans_std_bwd
                v_rot_target_bwd_norm = v_rot_target_bwd / rot_std_bwd

                v_trans_pred_bwd_norm = v_trans_pred_bwd / trans_std_bwd
                v_rot_pred_bwd_norm = v_rot_pred_bwd / rot_std_bwd

                loss_t_bwd = torch.mean((v_trans_target_bwd_norm - v_trans_pred_bwd_norm) ** 2, dim=-1)
                loss_r_bwd = torch.mean((v_rot_target_bwd_norm - v_rot_pred_bwd_norm) ** 2, dim=-1)

                loss_bwd = 2 * trans_weight * loss_t_bwd + rot_weight * loss_r_bwd
                protein_loss = loss_fwd.mean() + loss_bwd.mean()

                accum_loss += protein_loss.detach().item()
                accum_loss_trans_fwd += loss_t_fwd.mean().detach().item()
                accum_loss_rot_fwd += loss_r_fwd.mean().detach().item()
                accum_loss_trans_bwd += loss_t_bwd.mean().detach().item()
                accum_loss_rot_bwd += loss_r_bwd.mean().detach().item()

                optimizer.zero_grad()
                protein_loss.backward()

                if hasattr(args, 'grad_clip_norm') and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                optimizer.step()

    num_iters = len(loader) * args.train_number
    if num_iters > 0:
        accum_loss /= num_iters
        accum_loss_trans_fwd /= num_iters
        accum_loss_rot_fwd /= num_iters
        accum_loss_trans_bwd /= num_iters
        accum_loss_rot_bwd /= num_iters

        print(f"Current epoch loss: {accum_loss:.5f}")
        print(f"  Forward - Trans: {accum_loss_trans_fwd:.5f}, Rot: {accum_loss_rot_fwd:.5f}")
        print(f"  Backward - Trans: {accum_loss_trans_bwd:.5f}, Rot: {accum_loss_rot_bwd:.5f}")
    else:
        accum_loss = float('inf')
        print("No iterations were performed in this epoch")

    return accum_loss

def load_dataset(data_root, split, args):
    dataset_args = {
        "root": data_root,
        "split": split,
        "index_embed_size": args.t_emb_dim,
        "edge_embed_size": args.ipa_z_dim,
        "node_embed_size": args.ipa_s_dim,
        "coordinate_scaling": 0.1,
        "use_self_conditioning": getattr(args, 'use_self_conditioning', True),
        "num_bins": 22,
        "min_bin": 1e-5,
        "max_bin": 20.0,
        "time_interval": args.time_interval,
        "num_splits": args.num_splits
    }

    train_dataset = DatasetRigidSSLMD(**dataset_args)

    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        **dataloader_kwargs
    )

    return train_loader


def load_pretrained_weights(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        print(f"Pretrained model path {pretrained_path} not found.")
        return

    print(f"Loading pretrained model weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v

    model.load_state_dict(pretrained_dict, strict=False)

    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")


if __name__ == "__main__":
    print("MD Trajectory Pretraining")
    if not hasattr(args, 'use_self_conditioning'):
        args.use_self_conditioning = True
    if not hasattr(args, 'pretrained_weights'):
        args.pretrained_weights = ""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}

    if not hasattr(args, 'velocity_head_hidden'):
        args.velocity_head_hidden = args.ipa_s_dim
    if not hasattr(args, 'velocity_head_layers'):
        args.velocity_head_layers = 2

    model = model_setup()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights)

    if hasattr(args, 'velocity_head_lr_multiplier'):
        ipa_params = [p for n, p in model.named_parameters() if 'velocity_head' not in n]
        velocity_params = [p for n, p in model.named_parameters() if 'velocity_head' in n]

        model_param_group = [
            {"params": ipa_params, "lr": args.lr},
            {"params": velocity_params, "lr": args.lr * args.velocity_head_lr_multiplier}
        ]
        print(f"Using different learning rates - IPA: {args.lr}, Velocity heads: {args.lr * args.velocity_head_lr_multiplier}")
    else:
        model_param_group = [{"params": model.parameters(), "lr": args.lr}]

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model_param_group, lr=args.lr, weight_decay=args.decay, momentum=0.9)

    optimal_loss = 1e10

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print("Apply lr scheduler CosineAnnealingLR")

    start_epoch = 1
    if args.input_model_file != "":
        print(f"Resuming from checkpoint: {args.input_model_file}")
        pretrained_state_dict = torch.load(args.input_model_file)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dict["model"].items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")

        if "optimizer" in pretrained_state_dict:
            optimizer.load_state_dict(pretrained_state_dict["optimizer"])
            print("Restored optimizer state")
        if "scheduler" in pretrained_state_dict and lr_scheduler is not None:
            lr_scheduler.load_state_dict(pretrained_state_dict["scheduler"])
            print("Restored scheduler state")

        if "epoch" in pretrained_state_dict:
            start_epoch = pretrained_state_dict["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
        elif 'model_' in args.input_model_file:
            try:
                epoch_str = args.input_model_file.split('model_')[-1].split('.pth')[0]
                epoch_str = epoch_str.split('_')[0]
                start_epoch = int(epoch_str) + 1
                print(f"Resuming from epoch {start_epoch}")
            except:
                pass

    for epoch in range(start_epoch, start_epoch + args.epochs):
        accum_loss_total = 0
        print(f"Epoch: {epoch}")
        start_time = time.time()

        num_splits = args.num_splits
        split_indices = list(range(1, num_splits + 1))
        if hasattr(args, 'dataset_indices') and args.dataset_indices:
            split_indices = [int(x) for x in args.dataset_indices.split(',')]
        np.random.shuffle(split_indices)

        for split_idx in split_indices:
            print(f"Training on split {split_idx}/{num_splits}")
            train_loader = load_dataset(args.input_data_dir, str(split_idx), args)
            sub_accum_loss = train(model, device, train_loader, optimizer, args)
            accum_loss_total += sub_accum_loss
            del train_loader

        accum_loss_total /= len(split_indices)

        if accum_loss_total < optimal_loss:
            optimal_loss = accum_loss_total
            save_model(model, optimizer, lr_scheduler, save_best=True)

        print(f"MD Loss: {accum_loss_total:.5f}\tTime: {time.time() - start_time:.3f}s")

        if lr_scheduler is not None:
            lr_scheduler.step()
        print()

        save_model(model, optimizer, lr_scheduler, epoch=epoch, save_best=False)