# Global Conventions: Marker Localization and SLAM

## 1. Coordinate Frames
- **Camera Frame**:
  - `+x`: right
  - `+y`: down
  - `+z`: forward (optical axis)
- **World Frame**:
  - `+z`: up
  - Right-handed coordinate system.

## 2. Camera Intrinsics
- Resolution: VGA ($640 \times 480$), RGB, 8-bit.
- Pinhole camera model:
  - $f_x = 460.0$
  - $f_y = 460.0$
  - $c_x = 320.0$
  - $c_y = 240.0$
- Field of View: approximately $69.6^\circ$ horizontal FOV.
- Lens distortion: applied only during training augmentations and in specific stress-condition Blender renders. Canonical labels are always computed from the true 3D ray.

## 3. Landmark Measurement Representation
For a landmark located at camera-frame position $\mathbf{p} = (x, y, z)^\top$:
- **Azimuth** ($\theta$):
  $$\theta = \text{atan2}(x, z) \in [-\pi, \pi)$$
- **Elevation** ($\phi$):
  $$\phi = \text{atan2}(-y, \sqrt{x^2 + z^2}) \in [-\pi/2, \pi/2]$$
- **Inverse Depth** ($\rho$):
  $$\rho = \frac{1}{\|\mathbf{p}\|} = \frac{1}{\sqrt{x^2 + y^2 + z^2}} \quad (\text{in } \text{m}^{-1})$$
  *(Inverse range definition as specified in `specs.md`)*

## 4. Marker Specifications
- Marker is a planar square panel of fixed physical side length $0.25\text{ m}$.
- Visual pattern: identical custom high-contrast logo with an outer quiet white border.
- Operating depth range: $0.5\text{ m}$ to $6.0\text{ m}$ ($\rho \in [0.1667, 2.0]\text{ m}^{-1}$).
- Maximum viewing angle: $75^\circ$ between marker normal and view ray.
- Measurement target: exact logo center in 3D. Marker orientation is not used for measurement to eliminate PnP pose ambiguity.
