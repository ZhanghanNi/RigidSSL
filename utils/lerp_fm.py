import torch
import numpy as np

class LERP:
    '''
    Implement the linear interpolation in the R3 space.
    '''

    def __init__(self, config, stochastic_paths=False, coordinate_scaling=1.0):
        self.config = config
        self.stochastic_paths = stochastic_paths
        self.coordinate_scaling = coordinate_scaling
        self.g = config.get("g", 1.0)
        self.min_b = config.get("min_b", 0.1)
        self.max_b = config.get("max_b", 1.0)

    def _scale(self, x):
        return x * self.coordinate_scaling

    def _unscale(self, x):
        return x / self.coordinate_scaling

    def get_velocity(self, x0, x1):
        return x1 - x0

    def b_t(self, t):
        """Time-dependent coefficient for stochastic paths."""
        return self.min_b + t * (self.max_b - self.min_b)

    def forward_marginal(self, x0, x1, t):
        """Perform LERP with optional stochasticity."""
        x0, x1 = self._scale(x0), self._scale(x1)

        xt = (1 - t) * x0 + t * x1

        if self.stochastic_paths:
            noise = torch.randn_like(xt) * torch.sqrt(self.b_t(t))
            xt = xt + noise

        xt = self._unscale(xt)
        return {
            "trans_t": xt,
            }