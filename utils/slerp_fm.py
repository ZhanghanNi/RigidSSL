import torch
import numpy as np
from utils.iso3 import batch_sample
from utils.geometry import quat_to_rot, rot_to_quat

class SLERP:
    '''
    Implement the spherical linear interpolation in the SO3 space.
    '''

    def __init__(self, config, stochastic_paths=False):
        self.config = config
        self.stochastic_paths = stochastic_paths
        self.g = config.get("g", 1.0)
        self.min_sigma = config.get("min_sigma", 0.1)

    def slerp(self, q0, q1, t):
        """Performs SLERP between two quaternions q0 and q1."""
        q0 = q0 / torch.norm(q0)
        q1 = q1 / torch.norm(q1)

        dot = torch.dot(q0, q1)

        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.arccos(dot)

        qt = (torch.sin((1-t) * theta) * q0 + torch.sin(t * theta) * q1) / torch.sin(theta)

        return qt

    def forward_marginal(self, q0, q1, t):
        """Interpolates between two rotations using SLERP."""
        qt = self.slerp(q0, q1, t)
        rot_t = quat_to_rot(qt)

        if self.stochastic_paths:
            epsilon_t = self.compute_sigma_t(t)
            rot_t = batch_sample(rot_t, epsilon_t, 1)
            qt = rot_to_quat(rot_t)

        return {
            "qt": qt,
            }

    def compute_sigma_t(self, t):
        """Computes time-dependent noise standard deviation."""
        if isinstance(t, float):
            t = torch.tensor(t)
        return torch.sqrt(self.g**2 * t * (1 - t) + self.min_sigma**2)

