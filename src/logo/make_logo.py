"""
Procedural generator for custom logo and distractor markers.
Conforms to Milestone 1 specifications in specs.md.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np


def create_base_canvas(size: int = 1024) -> np.ndarray:
    """Create transparent 4-channel BGRA canvas."""
    return np.zeros((size, size, 4), dtype=np.uint8)


def draw_canonical_logo(
    size: int = 1024,
    accent_bgr: Tuple[int, int, int] = (10, 10, 240),  # Pure Vivid Red in BGR
    invert_bw: bool = False,
    flip_h: bool = False,
    flip_v: bool = False,
    inner_rotation_deg: float = 0.0,
    ring_dash: bool = False,
    blob_count: int = 4,
    omit_ring: bool = False,
    omit_accent: bool = False,
    swap_accent_color: Tuple[int, int, int] = None,
) -> np.ndarray:
    """
    Procedurally draws the high-contrast canonical logo or parametric variants.
    
    Design principles:
    - 1024x1024 BGRA with alpha.
    - High-contrast black/white core plus one saturated accent color (red wedge/triangle).
    - Strong low-frequency structure: thick outer ring, large asymmetric inner shape,
      distinctive blobs at different radii. No fine detail below 3% (~30px at 1024) so it survives at 15px.
    - Asymmetric under rotation and mirroring.
    - Unambiguous center point (small central dot + crosshairs marker).
    """
    img = create_base_canvas(size)
    cx, cy = size // 2, size // 2
    r_max = int(size * 0.46)  # Outer radius (~471 px)
    
    fg_color = (0, 0, 0, 255) if not invert_bw else (255, 255, 255, 255)
    bg_color = (255, 255, 255, 255) if not invert_bw else (0, 0, 0, 255)
    
    actual_accent = swap_accent_color if swap_accent_color is not None else accent_bgr
    accent_rgba = (actual_accent[0], actual_accent[1], actual_accent[2], 255)

    # 1. Base circular disk (White/Base)
    cv2.circle(img, (cx, cy), r_max, bg_color, -1, lineType=cv2.LINE_AA)

    # 2. Thick outer ring
    ring_thickness = int(size * 0.08)  # ~82 px, thick low-frequency
    if not omit_ring:
        if ring_dash:
            # Dashed ring (distractor variant)
            num_dashes = 12
            for i in range(num_dashes):
                if i % 2 == 0:
                    start_angle = i * (360 / num_dashes)
                    end_angle = (i + 1) * (360 / num_dashes)
                    cv2.ellipse(
                        img, (cx, cy), (r_max - ring_thickness // 2, r_max - ring_thickness // 2),
                        0, start_angle, end_angle, fg_color, ring_thickness, lineType=cv2.LINE_AA
                    )
        else:
            cv2.circle(img, (cx, cy), r_max - ring_thickness // 2, fg_color, ring_thickness, lineType=cv2.LINE_AA)

    # 3. Large asymmetric inner shape: a bold crescent/teardrop wedge
    # Rotated by inner_rotation_deg
    rad_offset = np.deg2rad(inner_rotation_deg)
    r_inner = int(r_max * 0.72)
    
    # Asymmetric polygonal hull in the top-left quadrant
    pts = []
    base_angles = [np.pi * 0.75, np.pi * 1.15, np.pi * 1.55, np.pi * 1.85, np.pi * 0.1]
    radii = [r_inner * 0.95, r_inner * 0.85, r_inner * 0.55, r_inner * 0.25, r_inner * 0.7]
    for a, r in zip(base_angles, radii):
        theta = a + rad_offset
        px = int(cx + r * np.cos(theta))
        py = int(cy + r * np.sin(theta))
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], fg_color, lineType=cv2.LINE_AA)

    # 4. Saturated accent wedge / polygon (Bottom-Right quadrant to contrast)
    if not omit_accent:
        accent_angles = [0.0, np.pi * 0.35, np.pi * 0.72]
        accent_radii = [r_inner * 0.95, r_inner * 1.0, r_inner * 0.75]
        pts_accent = []
        # Center anchor
        pts_accent.append([cx, cy])
        for a, r in zip(accent_angles, accent_radii):
            theta = a + rad_offset
            px = int(cx + r * np.cos(theta))
            py = int(cy + r * np.sin(theta))
            pts_accent.append([px, py])
        pts_accent = np.array(pts_accent, dtype=np.int32)
        cv2.fillPoly(img, [pts_accent], accent_rgba, lineType=cv2.LINE_AA)

    # 5. Distinctive blobs at different radii (low-frequency features, >= 35px)
    blob_configs = [
        # (radius_ratio, angle_rad, blob_radius_px, is_accent)
        (0.68, np.pi * 0.22, int(size * 0.045), False),   # ~46px blob
        (0.48, np.pi * 0.82, int(size * 0.050), True),    # ~51px accent blob
        (0.62, np.pi * 1.35, int(size * 0.038), False),   # ~39px blob
        (0.35, np.pi * 1.70, int(size * 0.042), False),   # ~43px blob
        (0.55, np.pi * 0.50, int(size * 0.035), False),   # optional 5th blob
    ]
    for idx in range(min(blob_count, len(blob_configs))):
        r_frac, a_rad, b_r, is_acc = blob_configs[idx]
        theta = a_rad + rad_offset
        bx = int(cx + (r_inner * r_frac) * np.cos(theta))
        by = int(cy + (r_inner * r_frac) * np.sin(theta))
        c = accent_rgba if (is_acc and not omit_accent) else fg_color
        cv2.circle(img, (bx, by), b_r, c, -1, lineType=cv2.LINE_AA)

    # 6. Unambiguous well-defined center point
    # White ring with black center dot + hairline cross (scaled for visibility)
    center_ring_r = int(size * 0.035)  # 35 px
    center_dot_r = int(size * 0.015)   # 15 px
    cv2.circle(img, (cx, cy), center_ring_r, bg_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), center_ring_r, fg_color, 2, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), center_dot_r, fg_color, -1, lineType=cv2.LINE_AA)

    # Apply mirror / flips if requested
    if flip_h:
        img = cv2.flip(img, 1)
    if flip_v:
        img = cv2.flip(img, 0)

    return img


def make_panel(
    logo_img: np.ndarray,
    panel_size: int = 1024,
    logo_fraction: float = 0.82,
    border_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Puts the logo on a square panel with a white quiet border.
    The physical panel represents 0.25 m side length.
    Returns RGB (3-channel) image.
    """
    panel = np.ones((panel_size, panel_size, 3), dtype=np.uint8)
    panel[:] = border_color
    
    target_logo_sz = int(panel_size * logo_fraction)
    resized_logo = cv2.resize(logo_img, (target_logo_sz, target_logo_sz), interpolation=cv2.INTER_AREA)
    
    offset = (panel_size - target_logo_sz) // 2
    
    alpha = resized_logo[:, :, 3] / 255.0
    for c in range(3):
        panel[offset:offset+target_logo_sz, offset:offset+target_logo_sz, c] = (
            alpha * resized_logo[:, :, c] +
            (1.0 - alpha) * panel[offset:offset+target_logo_sz, offset:offset+target_logo_sz, c]
        ).astype(np.uint8)
        
    return panel


def generate_distractors(size: int = 1024) -> Dict[str, np.ndarray]:
    """
    Generates 20 distinct hard-negative distractor logos:
    - Mirrored (H, V, HV)
    - Colour-swapped (Blue, Green, Yellow, Orange, Cyan)
    - Inverted B/W
    - Rotated inner structures (45, 90, 135, 180, 270)
    - Dashed outer rings
    - Altered blob configurations with significant area changes
    - Omitted features (missing ring, missing accent wedge)
    - Shape variations
    """
    distractors = {}
    
    # 1. Mirrored versions (subtle hard negatives: orientation/chirality flipped)
    distractors["distractor_01_mirrored_h"] = draw_canonical_logo(size, flip_h=True)
    distractors["distractor_02_mirrored_v"] = draw_canonical_logo(size, flip_v=True)
    distractors["distractor_03_mirrored_hv"] = draw_canonical_logo(size, flip_h=True, flip_v=True)
    
    # 2. Color swapped (using distinctly different saturated accents)
    distractors["distractor_04_blue_accent"] = draw_canonical_logo(size, swap_accent_color=(240, 60, 10))
    distractors["distractor_05_green_accent"] = draw_canonical_logo(size, swap_accent_color=(30, 240, 30))
    distractors["distractor_06_cyan_accent"] = draw_canonical_logo(size, swap_accent_color=(240, 240, 10))
    distractors["distractor_07_white_accent"] = draw_canonical_logo(size, swap_accent_color=(255, 255, 255))
    distractors["distractor_08_teal_accent"] = draw_canonical_logo(size, swap_accent_color=(180, 200, 20))
    
    # 3. Inverted black and white
    distractors["distractor_09_inverted_bw"] = draw_canonical_logo(size, invert_bw=True)
    distractors["distractor_10_inverted_bw_blue"] = draw_canonical_logo(size, invert_bw=True, swap_accent_color=(230, 80, 10))
    
    # 4. Rotated internal asymmetric features (45, 90, 135, 180, 270 degrees)
    distractors["distractor_11_rot45"] = draw_canonical_logo(size, inner_rotation_deg=45.0)
    distractors["distractor_12_rot90"] = draw_canonical_logo(size, inner_rotation_deg=90.0)
    distractors["distractor_13_rot135"] = draw_canonical_logo(size, inner_rotation_deg=135.0)
    distractors["distractor_14_rot180"] = draw_canonical_logo(size, inner_rotation_deg=180.0)
    distractors["distractor_15_rot270"] = draw_canonical_logo(size, inner_rotation_deg=270.0)
    
    # 5. Ring variations
    distractors["distractor_16_dashed_ring"] = draw_canonical_logo(size, ring_dash=True)
    distractors["distractor_17_no_ring"] = draw_canonical_logo(size, omit_ring=True)
    
    # 6. Combined geometric & structural alterations
    distractors["distractor_18_no_accent_wedge"] = draw_canonical_logo(size, omit_accent=True)
    distractors["distractor_19_dashed_blue"] = draw_canonical_logo(size, ring_dash=True, swap_accent_color=(230, 80, 10))
    distractors["distractor_20_mirrored_h_green"] = draw_canonical_logo(size, flip_h=True, swap_accent_color=(30, 220, 30))
    
    return distractors


def build_preview_sheet(
    canonical_logo: np.ndarray,
    distractors: Dict[str, np.ndarray],
    resolutions: List[int] = [64, 32, 16],
) -> np.ndarray:
    """
    Builds a preview sheet showing the canonical logo alongside distractors
    at full resolution and at low resolutions (64px, 32px, 16px).
    """
    # Grid: 21 items (canonical + 20 distractors).
    # Show each at 128px display size, with columns:
    # [Name, Full(128px), Res 64px, Res 32px, Res 16px]
    
    items = [("CANONICAL", canonical_logo)] + list(distractors.items())
    
    row_h = 140
    col_w = 140
    name_w = 260
    total_w = name_w + col_w * (1 + len(resolutions))
    total_h = row_h * (len(items) + 1)
    
    sheet = np.ones((total_h, total_w, 3), dtype=np.uint8) * 245
    
    # Header
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, "Item Name", (20, 40), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet, "Native", (name_w + 30, 40), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for idx, r in enumerate(resolutions):
        cv2.putText(sheet, f"{r}x{r} px", (name_w + col_w * (idx + 1) + 25, 40), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
    cv2.line(sheet, (0, 60), (total_w, 60), (180, 180, 180), 2)
    
    for row_idx, (name, logo) in enumerate(items):
        y_top = 70 + row_idx * row_h
        
        # Draw Name
        color = (0, 0, 180) if name == "CANONICAL" else (40, 40, 40)
        cv2.putText(sheet, name, (15, y_top + 70), font, 0.55, color, 2 if name == "CANONICAL" else 1, cv2.LINE_AA)
        
        # Alpha blend onto white for preview
        bgr = logo[:, :, :3]
        alpha = logo[:, :, 3] / 255.0
        blended = (alpha[:, :, None] * bgr + (1.0 - alpha[:, :, None]) * 255).astype(np.uint8)
        
        # 1. Native downscaled to 120x120 for display
        disp_native = cv2.resize(blended, (120, 120), interpolation=cv2.INTER_AREA)
        sheet[y_top+5:y_top+125, name_w+10:name_w+130] = disp_native
        
        # 2. Downscaled to test resolution, then upscaled with nearest-neighbor to 120x120
        for r_idx, r in enumerate(resolutions):
            low_res = cv2.resize(blended, (r, r), interpolation=cv2.INTER_AREA)
            up_disp = cv2.resize(low_res, (120, 120), interpolation=cv2.INTER_NEAREST)
            x_start = name_w + col_w * (r_idx + 1) + 10
            sheet[y_top+5:y_top+125, x_start:x_start+120] = up_disp
            
        cv2.line(sheet, (0, y_top + row_h - 10), (total_w, y_top + row_h - 10), (220, 220, 220), 1)
        
    return sheet


def generate_all_marker_assets(output_dir: str = "assets/logo") -> None:
    """Generate all canonical and distractor assets and save to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    distractors_dir = out_path / "distractors"
    distractors_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Canonical logo (1024x1024 BGRA)
    canonical = draw_canonical_logo(size=1024)
    cv2.imwrite(str(out_path / "canonical_logo.png"), canonical)
    
    # 2. Canonical panel (1024x1024 RGB, 0.25 m physical size)
    canonical_panel = make_panel(canonical, panel_size=1024)
    cv2.imwrite(str(out_path / "canonical_panel.png"), canonical_panel)
    
    # 3. 20 Distractors
    distractors = generate_distractors(size=1024)
    for name, dist_img in distractors.items():
        cv2.imwrite(str(distractors_dir / f"{name}.png"), dist_img)
        dist_panel = make_panel(dist_img, panel_size=1024)
        cv2.imwrite(str(distractors_dir / f"{name}_panel.png"), dist_panel)
        
    # 4. Preview sheet
    preview = build_preview_sheet(canonical, distractors, resolutions=[64, 32, 16])
    cv2.imwrite(str(out_path / "preview_sheet.png"), preview)
    print(f"Generated canonical logo, panel, {len(distractors)} distractors, and preview sheet in {out_path}")


if __name__ == "__main__":
    generate_all_marker_assets()
