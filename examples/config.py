import argparse
import os

parser = argparse.ArgumentParser()

# General
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)

# Training
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay", type=float, default=1e-4)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--num_workers", type=int, default=5)
parser.add_argument("--train_number", type=int, default=1)

# IPA architecture
parser.add_argument("--ipa_s_dim", type=int, default=256)
parser.add_argument("--ipa_z_dim", type=int, default=128)
parser.add_argument("--ipa_num_heads", type=int, default=8)
parser.add_argument("--ipa_num_blocks", type=int, default=4)
parser.add_argument("--ipa_coordinate_scaling", type=float, default=0.1)
parser.add_argument("--seq_tfmr_num_heads", type=int, default=4)
parser.add_argument("--seq_tfmr_num_layers", type=int, default=2)
parser.add_argument("--t_emb_dim", type=int, default=32)
parser.add_argument("--index_embed_size", type=int, default=32)
parser.add_argument("--aatype_embed_size", type=int, default=64)
parser.add_argument("--use_t", action="store_true", default=True)
parser.add_argument("--use_self_conditioning", action="store_true", default=True)

# Perturbation (Phase I)
parser.add_argument("--perturb_r", type=str, default="geodesic", choices=["geodesic", "random"])
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--trans_scale", type=float, default=0.03)
parser.add_argument("--perturb_r_max_angle", type=float, default=0.5)
parser.add_argument("--dataset_portion", type=str, default="full",
                    choices=["1percent", "10percent", "full"])

# Loss weights
parser.add_argument("--trans_loss_weight", type=float, default=1.0)
parser.add_argument("--rot_loss_weight", type=float, default=1.0)

# MD trajectory (Phase II)
parser.add_argument("--time_interval", type=int, default=1)
parser.add_argument("--num_splits", type=int, default=80)

# Data and checkpoints
parser.add_argument("--input_data_dir", type=str, default=os.environ.get("PERTURB_DATA_DIR", ""))
parser.add_argument("--output_model_dir", type=str, default=os.environ.get("OUTPUT_BASE_DIR", ""))
parser.add_argument("--input_model_file", type=str, default="")
parser.add_argument("--dataset_indices", type=str, default="", help="Comma-separated subset indices to train on, e.g. '2,5,10'")
parser.add_argument("--pretrained_weights", type=str, default="")

args = parser.parse_args()
