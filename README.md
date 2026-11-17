# Uncertainty Heads for Identical-Marker Localization and SLAM

This repository investigates single-pass negative log-likelihood (NLL) uncertainty heads (**Gaussian**, **Gaussian with $\beta$-NLL**, **Student-$t$**, and **Cauchy**) across three backbone architectures (**ResNet-9**, **ResNet-18**, and **ResNet-50**). Networks predict azimuth, elevation, and inverse depth ($\rho$) of visually identical 0.25 m planar markers for factor-graph localization and SLAM in GTSAM.

---

## 1. Quickstart & Pipeline Overview

The pipeline consists of modular milestones (M1–M8), each driven by standalone scripts and clean `make` targets.

```
       M1: Logo & Distractors (assets/logo/)
                     │
       M2: Synthetic Dataset Generation (OpenCV)
                     │
       M3: 2-Stage Model Training (checkpoints/)
                     │
       M4: Blender Validation Trajectories (data/blender/)
                     │
       M5: Per-Measurement & Calibration Evaluation
                     │
       M6: Factor-Graph Back-End (GTSAM Mode A & B)
                     │
       M7: Matrix Experiments (results.parquet)
                     │
       M8: Paper Plots & LaTeX Tables
```

---

## 2. Environment Setup

Activate your Python virtual environment (e.g. Conda, venv, or uv) containing PyTorch, GTSAM, OpenCV, NumPy, and Pandas:

```bash
# Verify required dependencies
python -c "import torch, cv2, numpy, pandas; print('Ready!')"
```

---

## 3. Step-by-Step Command Guide

### Step 1: Procedural Logo & Distractors Generation (M1)
Generates the canonical 0.25 m high-contrast asymmetric marker logo, panel texture, 20 hard-negative distractor markers, and a low-resolution survival preview sheet.

```bash
make logo
# Or via python directly:
python scripts/run_stage.py --stage logo
```
- **Output:** Assets saved to `assets/logo/canonical_logo.png`, `assets/logo/canonical_panel.png`, `assets/logo/distractors/`, and `assets/logo/preview_sheet.png`.

---

### Step 2: Synthetic Dataset Generation (M2)
Generates $640 \times 480$ synthetic images with procedural backgrounds (gradients, Perlin/value noise, room patches), lighting variations, motion/defocus blur, occlusions, and distractors.

```bash
# Generate rapid smoke dataset (200 images for fast testing)
make synth

# Or generate specific splits with custom counts:
python src/synth/generate.py --split train --num_images 10000 --workers 4 --out_dir data/synth/train
python src/synth/generate.py --split val   --num_images 1000  --workers 4 --out_dir data/synth/val
python src/synth/generate.py --split calib --num_images 1000  --workers 4 --out_dir data/synth/calib
```
- **Output:** Images and annotations saved in `data/synth/<split>/annotations.jsonl` and `annotations_compact.npz`.

---

### Step 3: Model Training (M3)
Trains models using the **Two-Stage Training Protocol**:
- **Stage 1 (Warm-up, 20% epochs):** Mean-only regression ($\Delta\text{azimuth}, \Delta\text{elevation}, \log\rho$) + CenterNet heatmap focal loss. Uncertainty branch is frozen.
- **Stage 2 (Joint Training, 80% epochs):** Uncertainty branch initialized from Stage 1 residual RMS and trained with the selected NLL loss.

#### Training Single Models
```bash
# Syntax: make train HEAD=<head> BACKBONE=<backbone> EPOCHS=<epochs> SEED=<seed>

# 1. Train Student-t with learned nu on ResNet-18
make train HEAD=student_t BACKBONE=resnet18

# 2. Train Cauchy on ResNet-9
make train HEAD=cauchy BACKBONE=resnet9

# 3. Train Gaussian beta-NLL on ResNet-50
make train HEAD=gaussian_beta BACKBONE=resnet50

# 4. Train with custom batch size and gradient accumulation (e.g. for 8GB GPUs):
python scripts/train.py --head student_t --backbone resnet18 --batch_size 8 --grad_accum 4 --epochs 30 --seed 42
```

Available `--head` options:
- `gaussian`: Standard Gaussian NLL ($\log\sigma + \frac{1}{2}(\frac{y-\mu}{\sigma})^2$).
- `gaussian_beta`: Gaussian $\beta$-NLL ($\beta=0.5$) preventing gradient-weighting collapse.
- `student_t`: Student-$t$ NLL with **learned** $\nu = 1.01 + \text{softplus}(\text{raw})$.
- `student_t_fixed`: Student-$t$ NLL with fixed $\nu = 3.0$ (heavy tails).
- `cauchy`: Cauchy NLL (Student-$t$ with fixed $\nu = 1.0$).
- `mse_fixed`: Fixed-covariance baseline (regression only).

Available `--backbone` options:
- `resnet9` (small DAWNBench style), `resnet18`, `resnet50`.

- **Output:** Checkpoints and config snapshots saved to `checkpoints/<backbone>_<head>_seed<seed>/model_final.pt`.

---

### Step 4: Blender Validation Trajectory Generation (M4)
Generates 6-DoF smooth trajectories with loop closures across procedural indoor rooms, generating ground-truth camera poses, landmark world coordinates, noisy odometry, and rendered frames.

```bash
make blender
# Or generate custom room / condition scenarios:
# Nominal condition
python src/blender/render.py --room room1_small_office --density medium --condition nominal --frames 300 --out_dir data/blender/room1_medium_nominal

# Stress conditions (motion blur, low light, occlusion, distractors)
python src/blender/render.py --room room1_small_office --density medium --condition motion_blur --frames 300 --out_dir data/blender/room1_medium_blur
python src/blender/render.py --room room1_small_office --density medium --condition low_light   --frames 300 --out_dir data/blender/room1_medium_lowlight
```
- **Output:** `poses_gt.npy`, `markers_gt.npy`, `odometry.npz`, `meas_gt.jsonl`, `manifest.json`, and images under `data/blender/<scenario>/`.

---

### Step 5: Per-Measurement & Calibration Evaluation (M5)
Evaluates any trained checkpoint against a validation dataset: computes detection AP, precision, recall, regression MAE/RMSE, Continuous Ranked Probability Score (CRPS), quantile coverage curves, post-hoc temperature recalibration, and inference FPS/latency.

```bash
# Evaluate a single checkpoint
python src/eval/evaluate.py \
    --checkpoint checkpoints/resnet18_student_t_seed42/model_final.pt \
    --data_dir data/blender/room1_medium_nominal \
    --out eval_resnet18_student_t.json

# Evaluate all 6 heads on a specific backbone (e.g. ResNet-18)
make eval-all-heads BACKBONE=resnet18

# Evaluate all 18 trained models (all heads x all backbones)
make eval-all
```

---

### Step 6: Factor-Graph Back-End & SLAM (M6)
Runs GTSAM factor graph optimization on the generated trajectories using custom $(\text{azimuth}, \text{elevation}, \rho)$ measurement factors with analytic Jacobians.

```bash
# Run a single backend localization trial (Mode B: Native IRLS, Mode A: Moment-matched)
python src/backend/slam.py --data_dir data/blender/room1_medium_nominal --mode B --head student_t --backbone resnet18

# Run backend optimization on all 6 heads for a specific backbone
make backend-all-heads BACKBONE=resnet18 MODE=B

# Run backend optimization on ALL 18 model combinations
make backend-all MODE=B
```
- **Output:** Reports Absolute Trajectory Error (ATE RMSE), accepted associations, and estimated poses.

---

### Step 7: Running the Experiment Matrix (M7)
Executes factor-graph runs across the matrix (heads $\times$ backbones $\times$ seeds $\times$ modes $\times$ odometry noise levels $\times$ conditions).
- **Caching & Resumability:** Individual trial runs are cached as JSON in `results_cache/`. Interrupted runs resume seamlessly.
- **Statistical Testing:** Automatically computes Two One-Sided Tests (TOST) equivalence between small backbones (ResNet-9 / 18) and ResNet-50, paired bootstrap 95% confidence intervals, and permutation tests for Head $\times$ Size interaction.

```bash
# 1. Rapid Smoke Matrix (fast run on a subset of heads, backbones, and noise levels)
make experiments-smoke
# or:
python scripts/run_experiments.py --smoke

# 2. Full Experiment Matrix
make experiments
# or:
python scripts/run_experiments.py --config configs/experiments.yaml
```
- **Output:** Single tidy DataFrame saved to `results.parquet`.

---

### Step 8: Multi-Model Benchmark & Plotting (RMSE, CRPS, Interval Score)
Automatically benchmarks all models in `checkpoints/` across heads and backbones on the validation dataset and produces both combined and separated comparisons:
- **Metrics Computed:** RMSE, CRPS, and Winkler Interval Score (90% central prediction interval).
- **Outputs:**
  - `validation_summary_table.md` (Markdown table)
  - `validation_summary_table.tex` (LaTeX table)
  - `validation_metrics_combined.png` (3x3 grid uniting all metrics and outputs)
  - `validation_metrics_azimuth.png` (Azimuth: RMSE, CRPS, IS)
  - `validation_metrics_elevation.png` (Elevation: RMSE, CRPS, IS)
  - `validation_metrics_inv_range_rho.png` (Inverse Range $\rho$: RMSE, CRPS, IS)

```bash
make benchmark
# Or via python directly:
python src/eval/benchmark_and_plot.py --checkpoints_dir checkpoints --val_data_dir data/blender/room1_medium_nominal --out_dir reports/validation_benchmark
```

---

### Step 9: Publication Figures & LaTeX Tables (M8)
Generates the publication figures and LaTeX `booktabs` tables directly from `results.parquet`:
- **Figure 3:** Per-measurement NLL and CRPS per output (nominal vs. stress conditions).
- **Figure 4:** Calibration curves (empirical vs. nominal coverage).
- **Figure 6:** Learned $\nu$ distribution of the Student-$t$ head vs. marker distance and environmental stress.
- **Figure 8:** Data association precision and recall vs. odometry drift-to-spacing ratio.
- **Figure 9:** Localization ATE RMSE and failure rate vs. odometry drift ratio.
- **Figure 12:** Accuracy vs. Compute Pareto curve (ATE RMSE vs. inference latency across backbones).
- **Figure 13:** Mode A (moment matching) vs. Mode B (native IRLS robust loss) comparison.
- **Table:** `reports/paper_figures/table_main_benchmark.tex` (`booktabs` formatted LaTeX table).
- **Document:** `docs/results_notes.md` detailing supported and unsupported scientific claims.

```bash
make paper-plots
# Or via python directly:
python src/plots/make_paper_figures.py --results results.parquet --out_dir reports/paper_figures
```

---

### Step 10: Real-Time Prediction & Localization Video Generator
Renders synchronized side-by-side MP4 videos demonstrating model prediction quality and factor-graph trajectory tracking:
- **Left Panel (Camera View):** Real-time image stream with predicted marker center detections, dynamic angular **uncertainty ellipses**, decoded $\rho$ (inverse depth), and learned $\nu$ tags.
- **Right Panel (Bird's-Eye View Map):** Top-down world map showing the ground truth path (gray), estimated camera trajectory (blue), moving camera heading arrow, and 3D landmarks.
- **Top HUD:** Real-time frame number, current ATE tracking drift (in cm), and detection counts.

```bash
# 1. Generate video for a single model (e.g. ResNet-18 Student-t, Mode B)
make video BACKBONE=resnet18 HEAD=student_t MODE=B

# Or specify a custom checkpoint path:
python src/plots/make_video.py \
    --checkpoint checkpoints/resnet18_student_t_seed42/model_final.pt \
    --data_dir data/blender/room1_medium_nominal \
    --out reports/videos/demo_resnet18_student_t.mp4

# 2. Batch generate videos for ALL trained checkpoints in checkpoints/
make video-all
```
- **Output:** MP4 videos saved in `reports/videos/`.

---

## 4. Running Unit Tests

Run the test suite to verify geometry round-trips, loss parity against `scipy.stats`, analytic Jacobians, and statistical metrics:

```bash
# Run all unit tests
make test
# Or individually:
python tests/test_m1_logo_and_geometry.py
python tests/test_m2_m3_losses_and_data.py
python tests/test_m4_m5_m6.py
python tests/test_m7_stats.py
```

---

## 5. Summary of Key Files

| Path | Purpose |
| :--- | :--- |
| `configs/experiments.yaml` | Matrix configuration (heads, backbones, seeds, noise levels, scenes). |
| `configs/train.yaml` | Two-stage training schedule, learning rates, and optimizer parameters. |
| `src/common/geometry.py` | Camera intrinsics, spherical measurement conversions, and 3D projections. |
| `src/logo/make_logo.py` | Procedural generation of 0.25 m canonical logo and 20 distractor panels. |
| `src/synth/generate.py` | Multi-process synthetic dataset generator. |
| `src/models/` | Stride-8 backbones (`resnet9`, `resnet18`, `resnet50`), CenterNet heads, decoders. |
| `src/losses/uncertainty_losses.py` | Gaussian NLL, $\beta$-NLL, Student-$t$ NLL, Cauchy NLL, focal loss. |
| `src/train/train.py` | Two-stage trainer with residual RMS bias initialization and weight EMA. |
| `src/blender/render.py` | 6-DoF trajectory and multi-room validation dataset renderer. |
| `src/eval/evaluate.py` | Evaluator computing AP, regression errors, CRPS, and calibration curves. |
| `src/eval/stats.py` | TOST equivalence testing, paired bootstrap CIs, and permutation tests. |
| `src/backend/factors.py` | GTSAM factor with analytic Jacobians, supporting Mode A and Mode B (IRLS). |
| `src/backend/slam.py` | Factor graph localization and SLAM runner. |
| `scripts/run_experiments.py` | Full matrix orchestrator outputting `results.parquet`. |
