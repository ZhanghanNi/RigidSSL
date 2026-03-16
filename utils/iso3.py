# Referred to FoldFlow: https://github.com/DreamFold/FoldFlow/blob/main/foldflow/models/so3_fm.py

import torch

def hat(v):
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def f_igso3_small(omega, sigma):
    """Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L99
    This function implements the approximation of the density function of omega of the isotropic Gaussian distribution.
    """


    eps = (sigma / torch.sqrt(torch.tensor([2])).to(device=omega.device)) ** 2

    pi = torch.Tensor([torch.pi]).to(device=omega.device)

    small_number = 1e-9
    small_num = small_number / 2
    small_dnm = (
        1 - torch.exp(-1.0 * pi**2 / eps) * (2 - 4 * (pi**2) / eps)
    ) * small_number

    return (
        0.5
        * torch.sqrt(pi)
        * (eps**-1.5)
        * torch.exp((eps - (omega**2 / eps)) / 4)
        / (torch.sin(omega / 2) + small_num)
        * (
            small_dnm
            + omega
            - (
                (omega - 2 * pi) * torch.exp(pi * (omega - pi) / eps)
                + (omega + 2 * pi) * torch.exp(-pi * (omega + pi) / eps)
            )
        )
    )

def _f(omega, eps):
    return f_igso3_small(omega, eps)

def _pdf(omega, eps):
    f_unif = angle_density_unif(omega)
    return _f(omega, eps) * f_unif

def _sample(eps, n):
        """
        Sample n points from IGSO3(I, eps) distribution.

        Args:
            eps: Concentration parameter for the distribution.
            n: Number of samples to generate.

        Returns:
            axis_angle: A tensor of shape (n, 3) representing the sampled axis-angle vectors.
        """
        num_omegas = 1024
        omega_grid = torch.linspace(0, torch.pi, num_omegas + 1).to(eps.device)[1:]  # skip omega=0

        # Numerical integration of (1 - cos(omega)) / pi * f_igso3(omega, eps) over omega
        pdf = _pdf(omega_grid, eps)
        dx = omega_grid[1] - omega_grid[0]
        cdf = torch.cumsum(pdf, dim=-1) * dx  # cumulative density function

        # Sample n points from the distribution
        rand_angle = torch.rand(n).to(eps.device)
        omegas = interp(rand_angle, cdf, omega_grid)

        # Sample rotation axes uniformly
        axes = torch.randn(n, 3).to(eps.device)
        axis_angle = omegas[..., None] * axes / torch.linalg.norm(axes, dim=-1, keepdim=True)

        return axis_angle

def batch_sample(mu, eps, n):
    """
    Generate batched samples from IGSO3 centered around mu using axis-angle representation.

    Args:
        mu: Mean rotation matrix in SO(3).
        eps: Concentration parameter for the distribution.
        n: Number of samples for each mean rotation in mu.

    Returns:
        A tensor of shape (mu.shape[0], 3, 3) containing rotation matrices.
    """
    samples = []
    for i in range(mu.shape[0]):
        aa_samples = _sample(eps[i], n).double()
        rot_samples = mu[i] @ so3_exp_map(aa_samples[i])
        samples.append(rot_samples)

    return torch.stack(samples)

def angle_density_unif(omega):
    return (1 - torch.cos(omega)) / torch.pi

def interp(x, xp, fp):
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])  # slope
    b = fp[:-1] - (m * xp[:-1])  # y-intercept

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), dim=1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]

def so3_exp_map(log_rot, eps = 0.0001):
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
        )

    return R