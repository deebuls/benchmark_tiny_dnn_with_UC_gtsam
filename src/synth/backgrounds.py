"""
Procedural background generator for synthetic dataset rendering.
Generates gradients, Perlin/value noise, polygons, stripes/checkers, and furniture/wall patches.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class ProceduralBackgroundGenerator:
    """Generates procedural backgrounds without requiring external image assets."""

    def __init__(self, is_validation: bool = False, seed: Optional[int] = None):
        self.is_validation = is_validation
        self.rng = np.random.default_rng(seed)

    def _generate_gradient(self, width: int, height: int) -> np.ndarray:
        c1 = self.rng.integers(0, 256, size=3, dtype=np.uint8)
        c2 = self.rng.integers(0, 256, size=3, dtype=np.uint8)
        angle = self.rng.uniform(0, 2 * np.pi)
        
        y, x = np.mgrid[0:height, 0:width]
        proj = x * np.cos(angle) + y * np.sin(angle)
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
        
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):
            bg[..., i] = (c1[i] * (1.0 - proj) + c2[i] * proj).astype(np.uint8)
        return bg

    def _generate_noise_field(self, width: int, height: int) -> np.ndarray:
        scale = self.rng.integers(8, 64)
        sh_h, sh_w = max(2, height // scale), max(2, width // scale)
        small_noise = self.rng.uniform(0, 255, size=(sh_h, sh_w, 3)).astype(np.float32)
        bg = cv2.resize(small_noise, (width, height), interpolation=cv2.INTER_CUBIC)
        return np.clip(bg, 0, 255).astype(np.uint8)

    def _generate_checker_stripes(self, width: int, height: int) -> np.ndarray:
        c1 = self.rng.integers(30, 220, size=3, dtype=np.uint8)
        c2 = self.rng.integers(30, 220, size=3, dtype=np.uint8)
        mode = self.rng.choice(["checker", "stripes_v", "stripes_h"])
        tile_size = self.rng.integers(20, 80)
        
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        y, x = np.mgrid[0:height, 0:width]
        
        if mode == "checker":
            mask = ((x // tile_size) + (y // tile_size)) % 2 == 0
        elif mode == "stripes_v":
            mask = (x // tile_size) % 2 == 0
        else:
            mask = (y // tile_size) % 2 == 0
            
        bg[mask] = c1
        bg[~mask] = c2
        return bg

    def _generate_room_like_patches(self, width: int, height: int) -> np.ndarray:
        # Base wall color
        wall_color = self.rng.integers(120, 240, size=3, dtype=np.uint8)
        bg = np.full((height, width, 3), wall_color, dtype=np.uint8)
        
        # Floor line
        floor_y = self.rng.integers(int(height * 0.55), int(height * 0.85))
        floor_color = self.rng.integers(40, 160, size=3, dtype=np.uint8)
        bg[floor_y:, :] = floor_color
        
        # Baseboard
        baseboard_h = self.rng.integers(10, 25)
        bb_color = self.rng.integers(20, 90, size=3, dtype=np.uint8)
        bg[floor_y:floor_y+baseboard_h, :] = bb_color
        
        # Random rectangular furniture / picture frames / shelves
        num_rects = self.rng.integers(1, 5)
        for _ in range(num_rects):
            rw = self.rng.integers(40, 200)
            rh = self.rng.integers(40, 200)
            rx = self.rng.integers(0, max(1, width - rw))
            ry = self.rng.integers(0, max(1, height - rh))
            rcolor = self.rng.integers(20, 220, size=3, dtype=np.uint8)
            cv2.rectangle(bg, (rx, ry), (rx + rw, ry + rh), [int(c) for c in rcolor], -1)
            
        return bg

    def generate(self, width: int = 640, height: int = 480) -> Tuple[np.ndarray, str]:
        # Choose background style; hold out room_like for validation or switch distribution
        if self.is_validation:
            choices = ["gradient", "noise", "room_like"]
            probs = [0.25, 0.25, 0.5]
        else:
            choices = ["gradient", "noise", "checker_stripes", "room_like"]
            probs = [0.3, 0.3, 0.2, 0.2]
            
        bg_type = self.rng.choice(choices, p=probs)
        if bg_type == "gradient":
            img = self._generate_gradient(width, height)
        elif bg_type == "noise":
            img = self._generate_noise_field(width, height)
        elif bg_type == "checker_stripes":
            img = self._generate_checker_stripes(width, height)
        else:
            img = self._generate_room_like_patches(width, height)
            
        return img, bg_type
