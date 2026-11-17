"""
Blender validation scene definitions, procedural room generation,
marker layout placements, smooth 6-DoF trajectories, and simulated odometry.
Can be executed directly via headless Blender `blender --background --python` or via the standalone renderer.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import numpy as np
from pathlib import Path


@dataclass
class RoomSpec:
    room_id: str
    width: float   # X size (m)
    depth: float   # Y size (m)
    height: float  # Z size (m)


ROOM_PRESETS = [
    RoomSpec("room1_small_office", 4.0, 5.0, 2.8),
    RoomSpec("room2_lab_narrow", 3.0, 8.0, 3.0),
    RoomSpec("room3_square_hall", 6.0, 6.0, 3.2),
    RoomSpec("room4_corridor_loop", 5.0, 9.0, 2.6),
    RoomSpec("room5_large_conference", 7.0, 8.0, 3.5),
]


def generate_marker_layout(
    room: RoomSpec,
    density: str = "medium",  # "dense", "medium", "sparse"
    marker_size: float = 0.25,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Places 0.25m markers on walls, furniture, and ceilings.
    Spacing: dense (~0.8m), medium (~1.5m), sparse (~3.0m).
    """
    rng = np.random.default_rng(seed)
    spacing = 0.8 if density == "dense" else (1.5 if density == "medium" else 3.0)

    markers = []
    m_id = 0

    # 1. Place on Walls
    # North Wall (y = room.depth)
    xs = np.arange(0.5, room.width - 0.5, spacing)
    zs = np.arange(1.0, room.height - 0.5, spacing)
    for x in xs:
        for z in zs:
            pos = [float(x), float(room.depth), float(z)]
            # Normal faces -Y (into the room)
            normal = [0.0, -1.0, 0.0]
            markers.append({"id": m_id, "position": pos, "normal": normal, "surface": "wall_north"})
            m_id += 1

    # South Wall (y = 0)
    for x in xs:
        for z in zs:
            pos = [float(x), 0.0, float(z)]
            normal = [0.0, 1.0, 0.0]
            markers.append({"id": m_id, "position": pos, "normal": normal, "surface": "wall_south"})
            m_id += 1

    # East Wall (x = room.width)
    ys = np.arange(0.5, room.depth - 0.5, spacing)
    for y in ys:
        for z in zs:
            pos = [float(room.width), float(y), float(z)]
            normal = [-1.0, 0.0, 0.0]
            markers.append({"id": m_id, "position": pos, "normal": normal, "surface": "wall_east"})
            m_id += 1

    # West Wall (x = 0)
    for y in ys:
        for z in zs:
            pos = [0.0, float(y), float(z)]
            normal = [1.0, 0.0, 0.0]
            markers.append({"id": m_id, "position": pos, "normal": normal, "surface": "wall_west"})
            m_id += 1

    # Ceiling (z = room.height) in dense mode
    if density == "dense":
        for x in xs[::2]:
            for y in ys[::2]:
                pos = [float(x), float(y), float(room.height)]
                normal = [0.0, 0.0, -1.0]
                markers.append({"id": m_id, "position": pos, "normal": normal, "surface": "ceiling"})
                m_id += 1

    return markers


def generate_6dof_trajectory(
    room: RoomSpec,
    num_frames: int = 600,
    with_loop_closure: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates smooth 6-DoF camera poses [N, 4, 4] (camera-to-world T_w_c).
    World frame: +Z up.
    Camera frame: +X right, +Y down, +Z forward.
    Includes loops that return to previously seen areas.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, num_frames)

    # Parametric trajectory: figure-8 or smooth Lissajous curve inside room boundaries
    margin = 1.0
    rx = (room.width - 2 * margin) / 2.0
    ry = (room.depth - 2 * margin) / 2.0
    cx = room.width / 2.0
    cy = room.depth / 2.0
    cz = 1.3  # Nominal handheld camera height

    if with_loop_closure:
        # Closes loop at t = 2*pi
        tx = cx + rx * np.sin(t)
        ty = cy + ry * np.sin(2 * t) / 2.0
        tz = cz + 0.15 * np.cos(3 * t)
    else:
        # Spiral outward
        tx = cx + (rx * (0.4 + 0.6 * t / (2 * np.pi))) * np.sin(2 * t)
        ty = cy + (ry * (0.4 + 0.6 * t / (2 * np.pi))) * np.cos(2 * t)
        tz = cz + 0.2 * np.sin(4 * t)

    poses = np.zeros((num_frames, 4, 4), dtype=np.float64)

    for i in range(num_frames):
        pos = np.array([tx[i], ty[i], tz[i]])
        
        # Tangent vector for forward direction (+Z in camera frame)
        if i < num_frames - 1:
            forward_world = np.array([tx[i+1] - tx[i], ty[i+1] - ty[i], tz[i+1] - tz[i]])
        else:
            forward_world = np.array([tx[i] - tx[i-1], ty[i] - ty[i-1], tz[i] - tz[i-1]])

        # Add gentle head oscillation (looking around)
        yaw_osc = 0.25 * np.sin(4 * t[i])
        pitch_osc = 0.1 * np.cos(3 * t[i])
        
        f_norm = np.linalg.norm(forward_world[:2]) + 1e-6
        base_yaw = np.arctan2(forward_world[1], forward_world[0]) + yaw_osc
        forward = np.array([np.cos(base_yaw), np.sin(base_yaw), pitch_osc])
        forward = forward / np.linalg.norm(forward)

        # World up is [0, 0, 1]
        world_up = np.array([0.0, 0.0, 1.0])
        # Camera right is forward x world_up
        cam_x = np.cross(forward, world_up)
        cam_x = cam_x / (np.linalg.norm(cam_x) + 1e-6)
        # Camera down is forward x cam_x
        cam_y = np.cross(forward, cam_x)
        cam_y = cam_y / (np.linalg.norm(cam_y) + 1e-6)
        # Camera forward (+Z) is forward
        cam_z = forward

        # Rotation matrix R_w_c: columns are camera axes in world frame
        R_w_c = np.column_stack([cam_x, cam_y, cam_z])

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_w_c
        T[:3, 3] = pos
        poses[i] = T

    return poses


def simulate_odometry(
    poses_gt: np.ndarray,
    noise_level: float = 0.01,  # e.g., 0.005, 0.01, 0.02, 0.05
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates relative between-pose odometry factors with realistic drift.
    Returns:
      - relative_poses_noisy: [N-1, 4, 4]
      - covariance_matrices: [N-1, 6, 6] (order: rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
    """
    rng = np.random.default_rng(seed)
    N = len(poses_gt)
    rel_noisy = np.zeros((N - 1, 4, 4), dtype=np.float64)
    covs = np.zeros((N - 1, 6, 6), dtype=np.float64)

    for i in range(N - 1):
        T1 = poses_gt[i]
        T2 = poses_gt[i + 1]
        
        # True relative pose in frame 1: T_1_2 = inv(T1) * T2
        T_1_2 = np.linalg.inv(T1) @ T2
        rel_pos = T_1_2[:3, 3]
        dist = float(np.linalg.norm(rel_pos))

        # Noise scales with distance
        sigma_trans = max(0.001, noise_level * dist)
        sigma_rot = max(np.deg2rad(0.1), np.deg2rad(1.0) * dist)

        # Corrupt translation
        trans_noise = rng.normal(0, sigma_trans, size=3)
        noisy_pos = rel_pos + trans_noise

        # Corrupt rotation (small angle perturbation)
        rot_noise_vec = rng.normal(0, sigma_rot, size=3)
        angle = np.linalg.norm(rot_noise_vec)
        if angle > 1e-6:
            axis = rot_noise_vec / angle
            # Rodrigues formula for delta R
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            delta_R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        else:
            delta_R = np.eye(3)

        noisy_R = T_1_2[:3, :3] @ delta_R

        T_noisy = np.eye(4, dtype=np.float64)
        T_noisy[:3, :3] = noisy_R
        T_noisy[:3, 3] = noisy_pos
        rel_noisy[i] = T_noisy

        # Covariance diagonal: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        cov_diag = np.array([sigma_rot**2, sigma_rot**2, sigma_rot**2, sigma_trans**2, sigma_trans**2, sigma_trans**2])
        covs[i] = np.diag(cov_diag)

    return rel_noisy, covs
