# Milestone 2 & 3 Implementation & Data Card

## Milestone 2: Synthetic Dataset Pipeline

### Data Generation Pipeline (`src/synth/`)
- **Generator (`src/synth/generate.py`)**:
  - Deterministic PRNG seeding per split and image.
  - Places 0 to 6 markers per image (~15% empty images for negative mining).
  - Uniform sampling in inverse-range $\rho \in [0.17, 2.0]$ ($0.5\text{ m}$ to $5.8\text{ m}$) and view-angle up to $75^\circ$.
  - Anti-aliased perspective homography warping (`cv2.warpPerspective`).
- **Procedural Backgrounds (`src/synth/backgrounds.py`)**:
  - Linear/angular directional gradients, cubic-interpolated value noise fields, checkerboards/stripes, and simulated room geometries (walls, floor, baseboards, furniture/shelves).
- **Realistic Variations & Augmentations (`src/synth/augmentations.py`)**:
  - Lighting: exposure, gamma, directional gradients across marker planes, specular glare blobs.
  - Occlusions: random polygons and ellipses covering $0-70\%$ of marker area.
  - Blur: motion blur (length up to 25 px, arbitrary angles), defocus Gaussian blur.
  - Sensor: Gaussian noise, vignetting, JPEG compression artifact simulation (quality 30 to 95).
  - Distractors: 0 to 4 hard-negative distractor logos per image (in 30% of frames).
- **Target Filtering & Rules**:
  - Ground truth positive target if visible fraction $\ge 0.30$ and center inside frame.
  - Severe occlusions/clipped markers mapped to an ignore mask (neither positive nor negative).
- **Output Formats**:
  - `annotations.jsonl`: Comprehensive per-marker metadata (3D camera position, azimuth, elevation, $\rho$, corners, pixel center, depth, view angle, visible fraction, difficulty tag, blur, exposure).
  - `annotations_compact.npz`: Compact indexed array archive for high-throughput PyTorch DataLoader batch streaming.

---

## Milestone 3: Model Architectures & Two-Stage Training

### Network Architecture (`src/models/`)
- Single forward pass on $640 \times 480$ VGA image producing a stride-8 output grid ($60 \times 80$ cells).
- **Backbones (`src/models/backbones.py`)**:
  - `resnet9`: DAWNBench-style 9-layer ResNet adapted with stride-8 downsampling.
  - `resnet18`: Standard torchvision ResNet-18 adapted to stride-8.
  - `resnet50`: Standard torchvision ResNet-50 adapted to stride-8.
  - Identical neck width ($128$ channels) across all backbones.
- **Dense CenterNet Uncertainty Head (`src/models/heads.py`)**:
  - Objectness logit (1 channel) initialized with prior bias $-2.19$ for focal loss stability.
  - Mean offsets (3 channels): $\Delta \theta$ (azimuth residual from cell-center ray), $\Delta \phi$ (elevation residual), and $\log(\rho)$.
  - Uncertainty parameters:
    - Gaussian / Gaussian $\beta$-NLL / Cauchy: $\log(\sigma)$ clamped to $[-7.0, 5.0]$.
    - Student-$t$: $\log(\sigma)$ and learned $\nu = 1.01 + \text{softplus}(\text{raw})$.
- **NMS & Measurement Decoding (`src/models/decode.py`)**:
  - $3 \times 3$ max-pooling non-maximum suppression on objectness heatmaps.
  - Decodes cell-relative azimuth and elevation back into absolute camera-frame angles and $\rho = \exp(\log\rho)$.

### Uncertainty Losses (`src/losses/uncertainty_losses.py`)
- Modified CenterNet focal loss on Gaussian-splatted heatmaps with ignore mask handling.
- `gaussian_nll_loss`: Standard Gaussian negative log-likelihood.
- `gaussian_beta_nll_loss`: $\beta$-NLL with $\beta = 0.5$ (Seitzer et al., 2022) to avoid gradient weighting collapse.
- `student_t_nll_loss`: Full Student-$t$ NLL parameterized by scale and $\nu$, verified against `scipy.stats.t.logpdf`.
- `cauchy_nll_loss`: Cauchy NLL (Student-$t$ with $\nu=1$), verified against `scipy.stats.cauchy.logpdf`.
- `smooth_l1_regression_loss`: Robust regression loss for Stage 1 warm-up.

### Two-Stage Trainer (`src/train/train.py`)
- **Stage 1 (Warm-up)**:
  - Trains backbone, neck, objectness, and mean heads ($\Delta\theta, \Delta\phi, \log\rho$) using focal loss + smooth L1.
  - Uncertainty branch is completely frozen and detached from gradients.
- **Stage 2 (Joint Training)**:
  - Uncertainty branch bias is initialized from the residual RMS ($\log(\text{residual RMS})$) computed from Stage 1 predictions on training data.
  - Uncertainty head is unfrozen.
  - Trains end-to-end with focal loss + the selected NLL head using AdamW, cosine annealing scheduler, weight EMA, and mixed precision.
