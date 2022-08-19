import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import cv2
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

camera_intrinsic = np.asarray([[320, 0, 320], [0, 320, 320], [0, 0, 1]], dtype = np.float32)
def make_one_pose(n_points):
    translation_gt = ((torch.randn([1,3])+5) * 100).unsqueeze(2)
    rotation_quaternion_gt = torch.randn(1,4)
    rotation_quaternion_gt = F.normalize(rotation_quaternion_gt, dim=-1)  # normalize to unit quaternion
    rotation_matrix_gt = quaternion_to_matrix(rotation_quaternion_gt)
    pose_extrinsics = torch.cat((rotation_matrix_gt,translation_gt),dim = -1)
    distCoeffs = np.asarray([0,0,0,0,0], dtype=np.float64)

    x3d = (torch.randn([n_points,3])+5) * 100
    x3d_h = torch.cat(((x3d, torch.ones(n_points,1))),dim = -1).unsqueeze(2)
    #print(pose_extrinsics.shape)
    #print(x3d_h.shape)
    x2d_h = torch.matmul(torch.from_numpy(camera_intrinsic),torch.matmul(pose_extrinsics, x3d_h))
    for i in range(n_points):
        x2d_h[i, 0] /= x2d_h[i, 2]
        x2d_h[i, 1] /= x2d_h[i, 2]
        x2d_h[i, 2] /= x2d_h[i, 2]
    x2d = x2d_h[:,:2].squeeze()

    x3d_np = np.array(x3d,dtype=np.float64)
    x2d_np = np.array(x2d,dtype=np.float64)
    _, rval, tval = cv2.solvePnP(x3d_np, x2d_np, camera_intrinsic, distCoeffs)
    rmat,_ = cv2.Rodrigues(rval)
    #print(rmat, tval, pose_extrinsics)
    return x3d_np,x2d_np,rotation_quaternion_gt,translation_gt.squeeze()




