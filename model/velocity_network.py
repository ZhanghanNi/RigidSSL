import torch
import math
from torch import nn
from torch.nn import functional as F
from model.ipa import IpaVelocity
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def calc_distogram(pos, min_bin=1e-5, max_bin=20.0, num_bins=22):
    """Calculate distance histogram for self-conditioning.

    Args:
        pos: [B, N, 3] atom positions
        min_bin: Minimum bin value
        max_bin: Maximum bin value
        num_bins: Number of bins

    Returns:
        [B, N, N, num_bins] distance histogram
    """
    if len(pos.shape) == 2:
        pos = pos.unsqueeze(0)  # [1, N, 3]

    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], dim=-1)[..., None]  # [B, N, N, 1]

    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).float()

    return dgram


class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions for distogram calculation.
                This should be the interpolated structure at time t.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []

        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        if self._embed_conf.embed_self_conditioning:
            sc_dgram = calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class VelocityNetwork(nn.Module):
    def __init__(self, model_conf):
        super(VelocityNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.ipa_model = IpaVelocity(model_conf)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats, direction='forward'):
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        node_embed, trans_output, rot_output = self.ipa_model(
            node_embed, edge_embed, input_feats, direction=direction)

        v_trans = trans_output
        v_rot = rot_output

        return node_embed, v_trans, v_rot