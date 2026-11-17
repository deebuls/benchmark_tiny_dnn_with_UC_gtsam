"""
Unit and acceptance tests for Milestone 1: Logo & Distractor Generation,
and Section 1: Geometry round trips.
"""

import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.geometry import (
    DEFAULT_INTRINSICS,
    point3d_to_measurement,
    measurement_to_point3d,
    point3d_to_pixel,
    pixel_to_bearing,
)
from src.logo.make_logo import (
    draw_canonical_logo,
    make_panel,
    generate_distractors,
)


class TestGeometry(unittest.TestCase):
    """Section 1 Acceptance Tests: Round-trip error below 1e-6."""

    def test_point3d_measurement_roundtrip(self):
        # Generate random 3D points in operating range: z in [0.5, 6.0], x and y within reasonable FOV
        np.random.seed(42)
        n_points = 1000
        z = np.random.uniform(0.5, 6.0, size=(n_points,))
        x = np.random.uniform(-1.5, 1.5, size=(n_points,)) * (z / 2.0)
        y = np.random.uniform(-1.0, 1.0, size=(n_points,)) * (z / 2.0)
        points_3d = np.stack([x, y, z], axis=-1)

        # Convert to measurement (azimuth, elevation, rho)
        meas = point3d_to_measurement(points_3d)

        # Convert back to 3D point
        recovered_3d = measurement_to_point3d(meas)

        # Check maximum absolute error
        max_err = np.max(np.abs(points_3d - recovered_3d))
        self.assertLess(max_err, 1e-6, f"Round trip error {max_err} exceeds 1e-6")

    def test_pixel_bearing_roundtrip(self):
        # Pixel coordinates inside image grid
        intrinsics = DEFAULT_INTRINSICS
        u = np.linspace(10, intrinsics.width - 10, 50)
        v = np.linspace(10, intrinsics.height - 10, 50)
        uu, vv = np.meshgrid(u, v)
        pixels = np.stack([uu, vv], axis=-1).reshape(-1, 2)

        # Pixel to bearing
        bearings = pixel_to_bearing(pixels, intrinsics)

        # Bearing is (dx, dy, dz) with dz > 0. Project back:
        # u_rec = fx * (dx / dz) + cx
        dx, dy, dz = bearings[:, 0], bearings[:, 1], bearings[:, 2]
        u_rec = intrinsics.fx * (dx / dz) + intrinsics.cx
        v_rec = intrinsics.fy * (dy / dz) + intrinsics.cy
        pixels_rec = np.stack([u_rec, v_rec], axis=-1)

        max_pixel_err = np.max(np.abs(pixels - pixels_rec))
        self.assertLess(max_pixel_err, 1e-6, f"Pixel roundtrip error {max_pixel_err} exceeds 1e-6")


class TestLogoM1(unittest.TestCase):
    """Milestone 1 Acceptance Tests: Logo distinctiveness, resolution survival, asymmetry."""

    def setUp(self):
        self.logo = draw_canonical_logo(size=1024)
        self.distractors = generate_distractors(size=1024)

    def test_logo_dimensions_and_channels(self):
        self.assertEqual(self.logo.shape, (1024, 1024, 4))
        # Ensure center pixel is dark/defined
        cx, cy = 512, 512
        # Center has dot
        self.assertEqual(self.logo[cy, cx, 3], 255)

    def test_asymmetry(self):
        # Must be asymmetric under rotation and mirror
        logo_bgr = self.logo[:, :, :3]
        logo_mirrored_h = cv2.flip(logo_bgr, 1)
        logo_mirrored_v = cv2.flip(logo_bgr, 0)
        logo_rot90 = cv2.rotate(logo_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Mean absolute pixel difference should be large (> 20)
        diff_h = np.mean(np.abs(logo_bgr.astype(float) - logo_mirrored_h.astype(float)))
        diff_v = np.mean(np.abs(logo_bgr.astype(float) - logo_mirrored_v.astype(float)))
        diff_rot = np.mean(np.abs(logo_bgr.astype(float) - logo_rot90.astype(float)))

        self.assertGreater(diff_h, 20.0, "Logo is too symmetric horizontally")
        self.assertGreater(diff_v, 20.0, "Logo is too symmetric vertically")
        self.assertGreater(diff_rot, 20.0, "Logo is too symmetric under 90 deg rotation")

    def test_distinctiveness_at_low_resolutions(self):
        """
        Acceptance: logo viewed at 16 px, 32 px and 64 px is visually distinct
        from all 20 distractors.
        """
        # Blend canonical onto white background
        alpha = self.logo[:, :, 3] / 255.0
        can_rgb = (alpha[:, :, None] * self.logo[:, :, :3] + (1.0 - alpha[:, :, None]) * 255).astype(np.float32)

        for res in [64, 32, 16]:
            can_low = cv2.resize(can_rgb, (res, res), interpolation=cv2.INTER_AREA)

            for d_name, d_img in self.distractors.items():
                d_alpha = d_img[:, :, 3] / 255.0
                d_rgb = (d_alpha[:, :, None] * d_img[:, :, :3] + (1.0 - d_alpha[:, :, None]) * 255).astype(np.float32)
                d_low = cv2.resize(d_rgb, (res, res), interpolation=cv2.INTER_AREA)

                # Compute normalized root-mean-square difference or mean absolute difference
                mae = np.mean(np.abs(can_low - d_low))
                # Distinctiveness threshold: even at 16px, distractor must differ meaningfully (MAE > 8 intensity levels)
                self.assertGreater(
                    mae, 8.0,
                    f"Distractor {d_name} is too similar to canonical logo at {res}x{res}px (MAE={mae:.2f})"
                )

    def test_panel_generation(self):
        panel = make_panel(self.logo, panel_size=1024, logo_fraction=0.82)
        self.assertEqual(panel.shape, (1024, 1024, 3))
        # Borders should be white
        self.assertTrue(np.all(panel[0:20, :, :] == 255))
        self.assertTrue(np.all(panel[-20:, :, :] == 255))
        self.assertTrue(np.all(panel[:, 0:20, :] == 255))
        self.assertTrue(np.all(panel[:, -20:, :] == 255))


if __name__ == "__main__":
    unittest.main()
