import os
import numpy as np


def load_cam_to_tcp(path: str) -> np.ndarray:
    """Load camera-to-TCP transform matrix from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Camera-to-TCP calibration file not found: {path}")
    return np.load(path)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert Flexiv [x,y,z,rw,rx,ry,rz] pose to 4x4 homogeneous transform."""
    x, y, z, rw, rx, ry, rz = pose
    qw, qx, qy, qz = rw, rx, ry, rz
    rotation = quaternion_to_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [x, y, z]
    return transform


def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform to Flexiv pose [x,y,z,rw,rx,ry,rz]."""
    qw, qx, qy, qz = quaternion_from_matrix(transform[:3, :3])
    x, y, z = transform[:3, 3]
    return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    qw, qx, qy, qz = quaternion
    norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
    if norm_sq < 1e-10:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm_sq
    x, y, z = qx, qy, qz
    wx, wy, wz = qw * x, qw * y, qw * z
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    return np.array(
        [
            [1 - scale * (yy + zz), scale * (xy - wz), scale * (xz + wy)],
            [scale * (xy + wz), 1 - scale * (xx + zz), scale * (yz - wx)],
            [scale * (xz - wy), scale * (yz + wx), 1 - scale * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def ensure_within_limits(vector: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """Clamp vector magnitude to max_norm."""
    norm = np.linalg.norm(vector)
    if norm > max_norm:
        return vector / norm * max_norm
    return vector
