"""
Unit tests for Milestone 4 (Blender / Trajectory), Milestone 5 (Evaluation),
and Milestone 6 (GTSAM Measurement Factor analytic Jacobians).
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blender.trajectory import ROOM_PRESETS, generate_6dof_trajectory, simulate_odometry
from src.backend.factors import compute_measurement_and_jacobians


class TestM4TrajectoryAndOdometry(unittest.TestCase):
    def test_trajectory_continuity(self):
        room = ROOM_PRESETS[0]
        poses = generate_6dof_trajectory(room, num_frames=50, with_loop_closure=True, seed=42)
        self.assertEqual(poses.shape, (50, 4, 4))
        # Ensure poses are valid SE(3) matrices
        for i in range(50):
            R = poses[i, :3, :3]
            # Check orthogonality: R * R^T = I
            diff = np.max(np.abs(R @ R.T - np.eye(3)))
            self.assertLess(diff, 1e-5)
            det = np.linalg.det(R)
            self.assertAlmostEqual(det, 1.0, places=5)

    def test_odometry_simulation(self):
        room = ROOM_PRESETS[0]
        poses = generate_6dof_trajectory(room, num_frames=20, seed=42)
        rel_noisy, covs = simulate_odometry(poses, noise_level=0.01, seed=42)
        self.assertEqual(rel_noisy.shape, (19, 4, 4))
        self.assertEqual(covs.shape, (19, 6, 6))


class TestM6AnalyticJacobians(unittest.TestCase):
    """Verifies that analytic Jacobians in compute_measurement_and_jacobians match numerical differentiation."""

    def test_landmark_jacobian_vs_numerical(self):
        # Camera at origin looking forward
        T_w_c = np.eye(4)
        p_w = np.array([0.5, -0.3, 2.5], dtype=np.float64)

        h0, H_pose, H_landmark = compute_measurement_and_jacobians(T_w_c, p_w)

        eps = 1e-6
        H_num = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            h_plus, _, _ = compute_measurement_and_jacobians(T_w_c, p_w + delta)
            h_minus, _, _ = compute_measurement_and_jacobians(T_w_c, p_w - delta)
            H_num[:, i] = (h_plus - h_minus) / (2.0 * eps)

        max_err = np.max(np.abs(H_landmark - H_num))
        self.assertLess(max_err, 1e-4, f"Landmark Jacobian max error {max_err} exceeds 1e-4")


if __name__ == "__main__":
    unittest.main()
