"""
Canonical geometry definitions and conversions for marker localization and SLAM.
Following specifications in docs/conventions.md.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 460.0
    fy: float = 460.0
    cx: float = 320.0
    cy: float = 240.0

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


# Default shared pinhole camera intrinsics
DEFAULT_INTRINSICS = CameraIntrinsics()

# Marker physical constants
MARKER_SIZE_METERS = 0.25  # Square marker side length in meters


def point3d_to_measurement(p: np.ndarray) -> np.ndarray:
    """
    Convert 3D camera-frame point p = (x, y, z) to (azimuth, elevation, rho).
    Camera frame: x right, y down, z forward.
    - azimuth: atan2(x, z) in [-pi, pi)
    - elevation: atan2(-y, sqrt(x^2 + z^2)) in [-pi/2, pi/2]
    - rho: 1 / ||p|| (inverse range in 1/m)
    """
    p = np.asarray(p, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    
    azimuth = np.arctan2(x, z)
    # Wrap azimuth to [-pi, pi)
    azimuth = (azimuth + np.pi) % (2.0 * np.pi) - np.pi
    
    r_xz = np.sqrt(x**2 + z**2)
    elevation = np.arctan2(-y, r_xz)
    
    norm_p = np.sqrt(x**2 + y**2 + z**2)
    rho = 1.0 / norm_p
    
    return np.stack([azimuth, elevation, rho], axis=-1)


def measurement_to_point3d(meas: np.ndarray) -> np.ndarray:
    """
    Convert (azimuth, elevation, rho) to 3D camera-frame point p = (x, y, z).
    - norm = 1 / rho
    - elevation phi = atan2(-y, sqrt(x^2 + z^2)) -> -y = norm * sin(phi) -> y = -norm * sin(phi)
    - r_xz = norm * cos(phi)
    - azimuth theta = atan2(x, z) -> x = r_xz * sin(theta), z = r_xz * cos(theta)
    """
    meas = np.asarray(meas, dtype=np.float64)
    azimuth, elevation, rho = meas[..., 0], meas[..., 1], meas[..., 2]
    
    norm = 1.0 / rho
    r_xz = norm * np.cos(elevation)
    
    x = r_xz * np.sin(azimuth)
    y = -norm * np.sin(elevation)
    z = r_xz * np.cos(azimuth)
    
    return np.stack([x, y, z], axis=-1)


def point3d_to_pixel(p: np.ndarray, intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS) -> np.ndarray:
    """
    Project 3D camera-frame point p = (x, y, z) to pixel coordinates (u, v).
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    """
    p = np.asarray(p, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    u = intrinsics.fx * (x / z) + intrinsics.cx
    v = intrinsics.fy * (y / z) + intrinsics.cy
    return np.stack([u, v], axis=-1)


def pixel_to_bearing(pixel: np.ndarray, intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS) -> np.ndarray:
    """
    Convert pixel coordinates (u, v) to unit bearing ray (dx, dy, dz) in camera frame.
    """
    pixel = np.asarray(pixel, dtype=np.float64)
    u, v = pixel[..., 0], pixel[..., 1]
    dx = (u - intrinsics.cx) / intrinsics.fx
    dy = (v - intrinsics.cy) / intrinsics.fy
    dz = np.ones_like(dx)
    ray = np.stack([dx, dy, dz], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    return ray / norm
