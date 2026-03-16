import numpy as np
import torch


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def compute_Inertia_Tensor(positions):
    mass_center = torch.mean(positions, dim=0, keepdim=True)
    positions = positions - mass_center

    eye = torch.eye(3).to(positions.device)
    inner_product = torch.einsum("bi,bi->b", positions, positions)
    outer_product = torch.einsum("bi,bj->bij", positions, positions)

    I_atom = inner_product.unsqueeze(1).unsqueeze(2) * eye.unsqueeze(0) - outer_product
    I = torch.mean(I_atom, dim=0)
    return I


def extract_rotation_matrix(positions):
    inertial_tensors = compute_Inertia_Tensor(positions)
    eigen_values, eigen_vectors = torch.linalg.eigh(inertial_tensors)
    rotation = eigen_vectors
    rotation = ensure_right_handedness(rotation)
    handness = determine_handness(rotation)
    assert handness == "Right-handed", "Rotation matrix should be right-hand."
    tie_index_list = check_tie_index(eigen_values)
    return rotation, tie_index_list


def ensure_right_handedness(eigenvectors):
    cross_product = torch.cross(eigenvectors[:, 0], eigenvectors[:, 1])
    if torch.dot(cross_product, eigenvectors[:, 2]) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]
    return eigenvectors


def determine_handness(vectors):
    determinant = np.linalg.det(vectors.cpu().numpy())
    if determinant > 0:
        return "Right-handed"
    else:
        return "Left-handed"


def check_tie_index(eigen_values):
    EPS = 1e-3
    tie_index_set = set()
    if abs(eigen_values[0] - eigen_values[1]) <= EPS:
        tie_index_set.add(0)
        tie_index_set.add(1)
    if abs(eigen_values[1] - eigen_values[2]) <= EPS:
        tie_index_set.add(1)
        tie_index_set.add(2)
    tie_index_list = sorted(list(tie_index_set))
    return tie_index_list
