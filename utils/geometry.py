import numpy as np
import torch

EPSILON = 1e-8


def LERP(translation_0, translation_1, timestep):
    return (1 - timestep) * translation_0 + timestep * translation_1


def SLERP(quaternion_0, quaternion_1, timestep):
    dot_product = torch.einsum("ab,ab->a", quaternion_0, quaternion_1).unsqueeze(1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    omega_t = omega * timestep
    sin_omega_t = torch.sin(omega_t)
    omega_1_t = omega * (1 - timestep)
    sin_omega_1_t = torch.sin(omega_1_t)
    quaternion_t = sin_omega_1_t / sin_omega * quaternion_0 + sin_omega_t / sin_omega * quaternion_1
    quaternion_t = torch.where(
        omega < EPSILON,
        LERP(quaternion_0, quaternion_1, timestep),
        quaternion_t,
    )
    mask = quaternion_t[..., 0] < 0
    quaternion_t[mask] *= -1
    return quaternion_t


def SLERP_derivative(quaternion_0, quaternion_1, timestep):
    dot_product = torch.sum(quaternion_0 * quaternion_1, dim=-1).unsqueeze(1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    omega = torch.acos(dot_product)
    omega_t = omega * timestep
    omega_1_t = omega * (1 - timestep)
    sin_omega = torch.sin(omega)

    cos_omega_t = torch.cos(omega_t)
    cos_omega_1_t = torch.cos(omega_1_t)

    derivative = omega / sin_omega * (-cos_omega_1_t * quaternion_0 + cos_omega_t * quaternion_1)
    return derivative


def SLERP_derivative_edge(quaternion_0, quaternion_1, timestep):
    dot_product = torch.sum(quaternion_0 * quaternion_1, dim=-1).unsqueeze(1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    omega = torch.acos(dot_product)

    use_lerp = omega < EPSILON
    lerp_derivative = quaternion_1 - quaternion_0

    omega_t = omega * timestep
    omega_1_t = omega * (1 - timestep)
    sin_omega = torch.sin(omega)

    cos_omega_t = torch.cos(omega_t)
    cos_omega_1_t = torch.cos(omega_1_t)

    slerp_derivative = omega / sin_omega * (-cos_omega_1_t * quaternion_0 + cos_omega_t * quaternion_1)

    derivative = torch.where(
        use_lerp.expand_as(quaternion_0),
        lerp_derivative,
        slerp_derivative
    )
    return derivative


# Quaternion-to-rotation conversion matrices
_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}

def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value
    return mat

_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    quat = quat[..., None] * quat[..., None, :]
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: torch.Tensor):
    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]] = rot

    K = [
        [R11 + R22 + R33, R32 - R23, R13 - R31, R21 - R12],
        [R32 - R23, R11 - R22 - R33, R12 + R21, R13 + R31],
        [R13 - R31, R12 + R21, R22 - R11 - R33, R23 + R32],
        [R21 - R12, R13 + R31, R23 + R32, R33 - R11 - R22],
    ]

    K = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in K], dim=-2)

    _, vectors = torch.linalg.eigh(K)
    quaternion = vectors[..., -1]

    mask = quaternion[..., 0] < 0
    quaternion[mask] *= -1

    return quaternion
