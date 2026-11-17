# Agent Spec: Uncertainty Heads for Identical-Marker Localization and SLAM

Build the full repository described below: synthetic data generation, training, Blender validation data, evaluation, factor-graph experiments in GTSAM, and paper plots. Work milestone by milestone. Every milestone has acceptance checks that must pass before moving on. Do not skip the unit tests.

## 0. Purpose and claim under test

Compare single-pass NLL uncertainty heads (Gaussian, Gaussian with beta-NLL, Student-t, Cauchy) on a network that outputs bearing and inverse-depth measurements of visually identical landmarks. Evaluate them in factor-graph localization (known map) and SLAM (unknown map), at three backbone sizes (ResNet-9, ResNet-18, ResNet-50).

Claim: for single-pass networks predicting azimuth, elevation and inverse depth of identical landmarks, the NLL head determines calibration, data-association reliability and factor-graph accuracy. Heavy-tailed heads let small networks match larger ones at lower compute.

The repo must produce evidence for both parts: the head effect, and the head x size interaction.

## 1. Global conventions (single source of truth, put in `src/common/geometry.py` and `docs/conventions.md`)

- Camera frame: x right, y down, z forward. World frame: z up.
- Image: VGA, 640 x 480, RGB, 8 bit. Pinhole intrinsics fx = fy = 460, cx = 320, cy = 240 (about 70 degree horizontal FOV). Keep these in one config and use the same values in OpenCV generation and Blender. Lens distortion is applied only as a training augmentation (labels always come from the true 3D ray); the Blender validation set uses an undistorted pinhole camera, plus a distorted variant as a stress condition.
- Measurement of a landmark at camera-frame position p = (x, y, z): azimuth = atan2(x, z); elevation = atan2(-y, sqrt(x^2 + z^2)); inverse depth rho = 1 / ||p|| (inverse range, in 1/m). Document this choice. All angles in radians, wrap azimuth to \[-pi, pi).
- Marker: one custom logo on a flat square panel of fixed physical size 0.25 m, identical in training, Blender and any real data. Apparent size is how the network infers inverse depth, so the size must never vary.
- Operating depth range: 0.5 m to 6 m (rho from 0.17 to 2.0). Maximum viewing angle between marker normal and view ray: 75 degrees.
- The marker measurement point is the logo centre. Orientation is not used (avoids PnP ambiguity).
- Frames are fixed in config; provide round-trip unit tests: project 3D point to pixel and back to bearing, and bearing + rho back to 3D, error below 1e-6.

## 2. Repository layout

```
repo/
  README.md, pyproject.toml, environment.yml, Makefile
  configs/            # yaml: data, model, train, eval, experiment matrix
  docs/               # conventions.md, data_card.md, results_notes.md
  assets/logo/        # generated logo files
  src/
    common/           # geometry, config loading, seeding, io
    logo/             # logo generator
    synth/            # OpenCV dataset generator
    data/             # torch datasets, train-time augmentation
    models/           # backbones, neck, heads, decode
    losses/           # gaussian, beta_nll, student_t, cauchy, huber
    train/            # two-stage trainer
    blender/          # bpy scene and trajectory generation, renderer
    eval/             # per-measurement metrics, calibration, detection
    backend/          # GTSAM factors, association, localization, slam
    baselines/        # classical template/feature baseline
    plots/            # all paper figures and LaTeX tables
  tests/
  scripts/            # one entry script per pipeline stage
```

Python 3.10+, PyTorch, OpenCV, NumPy, GTSAM (Python wrapper), Matplotlib, pandas, SciPy. Blender through the `bpy` module or headless Blender; if neither can be installed, still write the Blender scripts, document how to run them, and make every downstream stage consume a documented on-disk dataset format so Blender data can be dropped in.

## 3. Milestones

### M1. Custom logo

Generate the logo procedurally with OpenCV (`src/logo/make_logo.py`), 1024 x 1024 PNG with alpha, plus a 0.25 m square panel texture with a white quiet border.

Design requirements, chosen so a tiny network detects it easily:

- High-contrast black and white core plus one saturated accent colour (for example a red wedge), so it is detectable in grayscale and colour.
- Strong low-frequency structure: a thick outer ring, a large asymmetric inner shape, and 3 to 4 distinctive blobs at different radii. No fine detail below 3 percent of logo width, so it survives at 15 px.
- Asymmetric under rotation and mirror (so orientation is visible and a mirrored fake is a valid distractor).
- A well-defined centre point (the marker centre must be unambiguous at low resolution).

Output also a set of 20 distractor logos: mirrored, partially altered, colour-swapped, and similar-looking shapes. These are used as hard negatives.

Acceptance: logo viewed at 16 px, 32 px and 64 px is still visually distinct from all distractors; save a preview sheet.

### M2. Synthetic training dataset (OpenCV)

Generator `src/synth/generate.py`, multiprocessing, deterministic from a seed. Target: 200k training images, 10k synthetic validation images, 10k synthetic calibration images (held out for post-hoc recalibration), all at 640 x 480.

Pipeline per image:

1. Random camera-frame marker poses: 0 to 6 markers per image (about 15 percent of images with 0 markers). Sample depth, bearing, marker orientation (view angle up to 75 degrees), and roll. Ensure depth range and view-angle coverage are roughly uniform in rho and in angle.
2. Render the logo panel with a homography (cv2.warpPerspective with anti-aliasing) onto a background.
3. Backgrounds: procedural only (random gradients, Perlin or value noise, random polygons and lines, checker and stripe textures, blurred random colour fields, simulated wall, floor and furniture-like patches), plus an optional `--bg_dir` of user-supplied images. Keep a held-out set of background generators for validation.
4. Variations (all applied in the generator, each logged per image):
   - Lighting: global exposure and gamma, directional gradient across the panel, colour temperature shift, shadows cast across the panel, specular glare blobs and reflections.
   - Occlusion: random polygons and ellipses covering 0 to 70 percent of the marker; partial out-of-frame markers.
   - Blur: motion blur (length 0 to 25 px, random angle), defocus blur, both mixed.
   - Sensor: Gaussian and Poisson noise, JPEG compression (quality 30 to 95), vignetting, rolling-shutter shear (small), lens radial and tangential distortion (small to moderate, with true rays kept as labels).
   - Low resolution: markers near max depth and extreme view angles are naturally small or foreshortened. Do not filter them out; label their difficulty.
   - Distractors: 0 to 4 distractor logos per image (30 percent of images), and logo-like clutter. These have no label (hard negatives).
5. Annotation per image (JSON lines plus a compact npz), per marker: id within image, true 3D camera-frame position, azimuth, elevation, rho, pixel centre (undistorted and distorted), depth, view angle, visible fraction, occlusion fraction, blur length and angle, exposure and noise parameters, a `difficulty` tag (easy, medium, hard from a documented rule), and `in_frame` flags. Per image: all augmentation parameters, number of distractors, background type and seed.
6. A marker counts as a positive training target only if visible fraction is at least 0.3 and its centre is inside the image. Otherwise it is ignored in the loss (neither positive nor negative around its centre, using an ignore mask).

Acceptance:

- Visualization script draws labels onto 100 random images (centres, bearing, rho text) and the files are saved for manual inspection.
- Histograms of depth, view angle, occlusion, blur, and number of markers match the targets and are saved as figures.
- Label sanity unit test: reproject stored 3D points through the camera model (including distortion) and match the stored pixel centre within 0.5 px.

### M3. Models and training

**Architecture** (`src/models/`): backbone + light neck + dense head, single forward pass, stride-8 output grid (80 x 60 cells at VGA).

- Backbones: ResNet-9 (small ResNet in the DAWNBench style, adapted to stride 8), ResNet-18, ResNet-50 (user wrote 54: treat as ResNet-50, make the variant configurable). All adapted with a small FPN-like neck to produce a stride-8 feature map. Keep neck and head widths identical across backbones.
- Head outputs per cell: objectness logit (1); mean offsets for azimuth, elevation (residuals from the cell-centre ray, decoded to absolute angles) and rho (predicted in log space and decoded); uncertainty parameters per output (3 outputs).
- Uncertainty parametrisation per head type: Gaussian: log-scale (sigma). Student-t: log-scale and nu = 1 + softplus(raw) (learned), with a fixed-nu variant (nu = 3 and nu = 5). Cauchy: log-scale only (Student-t with nu fixed to 1). Clamp log-scale to a documented range to avoid collapse.
- Independent outputs (diagonal). Provide a full-covariance Gaussian head only as an optional ablation (disabled by default).
- Inference: 3 x 3 max-pool non-maximum suppression on the objectness heatmap, threshold, decode measurements and uncertainty at peaks.

**Heads to implement** (`src/losses/`):

1. `mse_fixed`: mean regression only, fixed covariance estimated from the calibration set (lower-bound baseline).
2. `gaussian`: standard Gaussian NLL.
3. `gaussian_beta`: beta-NLL (beta = 0.5) to fix the known gradient-weighting problem, so Gaussian is not a straw man.
4. `student_t`: Student-t NLL with learned nu (and fixed-nu ablation).
5. `cauchy`: Cauchy NLL. All NLLs are per output (azimuth, elevation, rho), summed. Losses must be numerically stable and unit-tested against scipy.stats log-pdfs.

**Objectness loss**: focal loss on a Gaussian-splatted heatmap (CenterNet style) with the ignore mask. Regression and NLL terms apply only at positive centre cells (optionally also the 8 neighbouring cells with lower weight, configurable).

**Two-stage training** (`src/train/train.py`):

- Stage 1 (mean-only warm-up, N1 iterations, default 20 percent of the schedule): train backbone, neck, objectness and mean outputs (azimuth, elevation, rho) with a robust regression loss (smooth-L1). The uncertainty branch is frozen and not in the loss. Rationale: NLL training from scratch is unstable and the mean should be good before variance is learned.
- Stage 2 (joint, remaining iterations): initialise the uncertainty branch bias from stage-1 residual statistics on a data subset (log of residual RMS per output), unfreeze the whole network, and train with objectness + the selected NLL for the three outputs. Learning-rate warm-up on re-start, cosine decay, lower LR for the backbone than for heads.
- Note on interpretation: the request said to first train only the outputs. This spec interprets that as training the mean outputs (with the backbone) before enabling the uncertainty head. If the intention was to freeze the backbone in stage 1, add a config flag `freeze_backbone_stage1` and run it as an ablation.

**Train-time augmentation** (`src/data/augment.py`): photometric only unless labels are updated. Colour jitter, brightness and contrast, gamma, additional Gaussian blur and noise, JPEG, random erasing (update the visible-fraction mask), cutout occluders, and a roll rotation about the principal point (up to 15 degrees) that also rotates the ray labels analytically. No horizontal flip (the logo is asymmetric and the real logo is not mirrored) and no crop or rescale (these change effective intrinsics and the rho-from-size cue).

**Other training details:** AdamW, mixed precision, EMA of weights, gradient clipping, logging of loss terms, learned nu and mean scale per epoch, fixed seeds, checkpoint at the end of each stage, identical protocol across backbones and heads with a small per-head learning-rate search done with equal effort on the synthetic validation set. Every configuration trains with 3 seeds. Training budget in config; print estimated GPU hours before launching the full matrix.

Acceptance:

- Overfit test: 200 images, loss goes near zero in a few hundred steps for every head and backbone.
- A short run on each head reaches heatmap AP above 0.9 on easy synthetic validation images and shows non-degenerate learned scales (no variance collapse, nu within range).
- Unit tests on loss gradients and finite values with extreme residuals.

### M4. Blender validation dataset

Scripts in `src/blender/`. The same logo and the same physical size (0.25 m) are used. Set the Blender camera to the shared intrinsics (sensor and focal length derived from fx, cx, cy), render at 640 x 480. Use textures, materials, lighting and furniture that are not seen by the synthetic training generator.

Scenes:

- At least 5 procedurally generated rooms of different sizes, with walls, floor, furniture and clutter. Hold some textures and a few rooms as the main test set.
- Marker layouts: dense, medium and sparse (for example 0.8 m, 1.5 m and 3 m typical nearest-neighbour spacing), markers on walls, furniture and ceilings, all with the identical logo.
- Trajectories: at least 4 per room, 6-DoF smooth handheld or robot-like paths of 500 to 1000 frames, including loops that return to previously seen areas (for loop closure through identical landmarks) and path segments with fast rotation.
- Rendering variants per trajectory (same poses, different rendering): nominal, low light, strong lighting contrast and glare, motion blur (Blender motion blur from real velocity), partial occlusion by moving or static objects, distractors (posters and logo-like objects, mirrored logos), camera lens distortion and noise.
- Output format: one folder per scene and condition: images, `poses_gt.npy` (camera-to-world), `markers_gt.npy` (world positions and ids; ids used only for evaluation and the oracle), `meas_gt.jsonl` (per-frame per-marker true azimuth, elevation, rho, visibility fraction, occlusion, in-frame), camera intrinsics file, and a `manifest.json` describing scene, layout, condition, seed.
- Also compute and store simulated odometry per trajectory (see M6) as separate files with a fixed seed, so that all models see identical odometry noise.

Acceptance:

- Ground-truth bearing check: reproject `markers_gt` through `poses_gt` and the camera model and confirm they hit the logo in a sample of rendered frames (overlay images saved).
- Report dataset statistics per condition (marker counts, depth, view angle, visibility).

### M5. Per-measurement evaluation

Run each trained model on all Blender frames (and the synthetic validation set) and match predicted peaks to ground-truth markers (nearest in angle within a gate, one-to-one) to compute:

- Detection: precision, recall, AP, false positives per frame, miss rate, split by condition, distance bin and visibility bin.
- Regression accuracy per output (azimuth, elevation, rho separately): MAE, RMSE, median error, error tails (95th and 99th percentile).
- Uncertainty quality per output and jointly: mean NLL (from each head native log-likelihood, directly comparable across heads since all are in measurement units), CRPS (closed form for Gaussian, numeric for Student-t and Cauchy), quantile coverage curves (nominal vs empirical for central intervals using native quantiles), PIT histograms, and sharpness (median predicted interval width). Do not use variance-based ECE for heavy-tailed heads because Cauchy has no variance and Student-t has none for nu at most 2.
- Post-hoc recalibration check: fit a single temperature (scale multiplier per output) on the synthetic calibration set, report all uncertainty metrics before and after. This shows whether the heads are intrinsically calibrated or just need rescaling, and whether the small models gain more from rescaling.
- Error vs predicted scale scatter plots to show whether the predicted uncertainty is informative (Spearman correlation between predicted scale and absolute error).
- Repeat all metrics on each condition (nominal and stress) and with results aggregated across training seeds with confidence intervals.
- Cost: parameters, FLOPs, memory, and latency in PyTorch and, if possible, ONNX or TensorRT, on the dev GPU, CPU, and a target embedded device if available (documented as optional if not). Use identical VGA input.

### M6. Factor-graph back-end (GTSAM)

Implement `src/backend/`. Poses are Pose3 and landmarks are Point3 (the measurements use the logo centre only).

**Measurement factor.** Predicted measurement h(pose, landmark) = (azimuth, elevation, rho) as defined in section 1, with azimuth residual wrapped. Implement as one scalar factor per output (three scalar factors per observation sharing the same variables) so the total loss equals the sum of the head independent per-output negative log-likelihoods. Use analytic Jacobians and validate against numerical differentiation.

**How each head becomes a factor** (this is the core of the study; implement both modes and run both):

- Mode A, Gaussian moment-matching: each head is converted to a Gaussian noise model. Gaussian head uses its sigma. Student-t with nu above 2 uses scale times sqrt(nu / (nu - 2)). Cauchy has no variance, so use a documented quantile-matching rule (match the central interquartile range of the Cauchy to a Gaussian). Document the rule and test its effect with an ablation of one alternative (match 90 percent interval).
- Mode B, native heavy-tailed likelihood: the scalar residual is whitened by the head scale and passed through a robust loss that equals the head NLL up to a constant. Student-t: loss(r) = ((nu + 1) / 2) \* log(1 + r^2 / nu), IRLS weight w(r) = (nu + 1) / (nu + r^2). Cauchy: nu = 1. Gaussian: standard least squares. Implement via `gtsam.noiseModel.Robust` with `mEstimator.Custom` (custom weight and loss functions) if available in the installed Python wrapper. If `Custom` is not exposed in the Python bindings, implement the factor in C++ with a pybind or use GTSAM CustomFactor with an outer IRLS loop that re-weights per iteration. Because nu and scale differ per measurement, a new noise model instance is created per measurement. A unit test must verify that the implemented loss and weight match the head NLL (up to constant) on a grid of residuals and nu values, and that the optimizer reduces the robust cost.
- Also provide `fixed_cov` (mse\_fixed head) and `gaussian_beta` through Mode A and Mode B (the latter is simply Gaussian).

**Odometry.** Between-factors from ground-truth relative poses corrupted with noise (translation noise as a fraction of travelled distance, rotation noise in degrees per metre and per radian turned, with a small bias random walk). Provide at least 4 noise levels (for example 0.5 percent, 1 percent, 2 percent, 5 percent of distance). Generate and cache with fixed seeds, as in M4. Always keep odometry independent of images so that all heads and sizes see the same inputs.

**Data association** (identical landmarks, so geometry only):

- Localization with known map: for each detection, predict candidate landmarks from the current pose estimate (propagate pose marginal covariance through the Jacobian and inflate the head scale in quadrature). Compute a likelihood-based gate score per head native likelihood: accept a candidate if the negative log-likelihood ratio relative to the mode is below a threshold. The gate threshold is calibrated per head (and per size) from the 99th percentile of the score of correct matches on the synthetic calibration set (documented), never tuned on the Blender test set. Use nearest-neighbour with gating and, optionally, a joint compatibility or Hungarian assignment among the detections of one frame (one-to-one constraint). Detection with no candidate in the gate is dropped. Ambiguous (two or more candidates inside the gate) are either rejected or handled by a mixture, controlled by a flag; report both policies.
- SLAM with unknown map: same gating against existing landmarks; observations with no match start a tentative landmark, initialised by triangulation from bearing and rho with covariance; a tentative landmark is promoted after at least 3 consistent observations; landmark merging step to resolve duplicates when two landmarks are within a compatibility gate. Document rules and thresholds in config.
- Optimiser: iSAM2 for online runs and Levenberg-Marquardt batch for final refinement. Record both online and final results.
- Oracle runs: oracle association (ground-truth ids) for both tasks as an upper bound.
- Fair-comparison rule: the same association, same odometry, same optimiser, and same thresholds-calibration procedure for all heads and sizes.
- Divergence detection: define failure as ATE final above a threshold (for example 1 m or 10 percent of trajectory length, set in config) or optimiser failure, and count it.

**Classical non-learned baseline** (`src/baselines/`): detect the logo by multi-scale template matching and feature matching (SIFT or ORB plus homography) in the VGA image, compute the marker centre and rho from the matched homography and the known marker size, and propagate a covariance from a simple corner-noise model. Feed it through the same back-end. This shows what the networks add.

### M7. Experiments (`scripts/run_experiments.py`, driven by `configs/experiments.yaml`)

Experiment matrix:

- Heads: mse\_fixed, gaussian, gaussian\_beta, student\_t (learned nu), student\_t fixed nu=3, cauchy.
- Sizes: ResNet-9, ResNet-18, ResNet-50.
- Seeds: 3 training seeds (more if budget allows).
- Back-end modes: A (Gaussian moment-matched) and B (native heavy-tailed).
- Tasks: localization with known map, SLAM.
- Scenes: all Blender rooms, marker layouts (dense, medium, sparse), odometry noise levels, rendering conditions (nominal and stress).
- Association: gated nearest neighbour (main), oracle (upper bound).
- Baselines: classical detector, and (optional, clearly labelled out of scope) a small ensemble of 5 ResNet-9 Gaussian models as a multi-pass reference.

Key sweeps:

1. Ratio of odometry drift to marker spacing (the main aliasing stressor).
2. Rendering condition (nominal vs blur, low light, occlusion, distractors, distortion).
3. Marker density.
4. With and without loop closures (SLAM).

Metrics per run:

- Localization: ATE (RMSE after no alignment for known map, and with SE(3) alignment reported separately for SLAM), RPE (translation and rotation), final drift, per-frame error, failure flag.
- SLAM: ATE after alignment, landmark map RMSE after alignment, number of landmarks vs ground truth, number of duplicate and missed landmarks.
- Association: precision, recall, number of accepted wrong associations, missed associations, rate of ambiguous rejections.
- Consistency: NEES of poses (and landmarks in SLAM) against chi-square bounds, NIS of measurement residuals, per-output independence check (correlation of the three residual channels).
- Success rate, median, 90th percentile and worst-case error across seeds and trajectories (failures are bimodal, do not report only means).
- Runtime of the back-end and of the network per frame.

Statistics (`src/eval/stats.py`): paired bootstrap confidence intervals over trajectories, paired by trajectory and odometry seed; define an equivalence margin in config (ATE within X percent and failure rate within Y points of ResNet-50) and test comparability with TOST or paired bootstrap, separately for each head. Include a test of whether the head x size interaction is significant (for example a mixed-effects or two-way permutation test on ATE and failure rate).

Every run writes a row to a single tidy `results.parquet` with all config fields (head, size, seed, mode, task, scene, layout, condition, odometry level, association, metrics). Runs are resumable and cached.

### M8. Plots and tables for the paper (`src/plots/`)

All figures come from `results.parquet` with one script and a fixed style (vector PDF, readable at column width, colour-blind-safe palette, consistent colour per head and marker per size). Generate:

1. Example images: logo, synthetic training samples with variations, Blender validation frames, with predicted detections and uncertainty ellipses.
2. Dataset statistics (depth, view angle, occlusion, blur histograms) for training vs Blender.
3. Per-measurement NLL and CRPS per output, head x size (bar plots with confidence intervals), nominal vs stress.
4. Calibration: quantile coverage curves and PIT histograms per head, per output, plus before and after temperature rescaling.
5. Predicted scale vs absolute error scatter and binned reliability plots, per head.
6. Learned nu distribution of the Student-t head versus condition and distance.
7. Detection AP and false positives per frame vs condition.
8. Association precision, recall and accepted wrong-association counts vs odometry drift to spacing ratio, per head, with oracle line.
9. Localization ATE and failure rate vs drift to spacing ratio, per head and size, both back-end modes.
10. SLAM ATE and map error vs drift to spacing ratio and vs condition, per head and size.
11. NEES consistency plots (median and fraction inside chi-square bounds) per head.
12. Accuracy vs compute Pareto plot: ATE and failure rate against parameters, FLOPs and latency, with each head as a separate curve over the three sizes (the central figure for the claim), including the classical baseline.
13. Mode A vs Mode B comparison: gain from using the native heavy-tailed likelihood in the back-end.
14. Example trajectories (best, median, failed runs) with estimated vs ground-truth path and map.
15. Equivalence test summary: for each head, whether ResNet-9 and ResNet-18 are within the equivalence margin of ResNet-50. Also produce LaTeX tables (booktabs) of the main numbers (mean and 95 percent confidence interval), and a `docs/results_notes.md` that lists the key findings and any claim that is NOT supported by the data.

## 4. Additional steps added to the plan (not in the original request)

- Fixed-covariance and Gaussian beta-NLL baselines so the Gaussian head is not a weak reference.
- Post-hoc temperature recalibration on a held-out synthetic calibration set, to separate intrinsic calibration from simple rescaling.
- Both back-end modes (moment-matched Gaussian vs native heavy-tailed likelihood), because the study must show whether gains come from better calibration or from the likelihood used in the optimiser.
- Likelihood-based association gates calibrated per head on synthetic data only, never on test data.
- Oracle-association and classical-detector baselines, odometry noise sweeps, and drift to spacing ratio sweeps.
- Detection metrics (false positives, misses) reported separately because they interact with association.
- Failure-rate and worst-case reporting, multi-seed training, paired bootstrap confidence intervals and an equivalence test.
- Hold-out discipline: Blender textures, rooms and lighting never used for training or threshold tuning; synthetic calibration split separate from the validation split.
- Compute reporting (parameters, FLOPs, latency) for each size.
- A real-world sequence is not part of this build, but the data format and the evaluation scripts must accept a real sequence (images, intrinsics, and an optional motion-capture or total-station reference). Write `docs/real_data_howto.md`. Real-world data will be needed before submission to a robotics venue, since calibration claims are the first to fail under a sim-to-real gap.
- Reproducibility: seeds everywhere, config snapshots saved with each run, environment file, a Makefile with targets for each stage (`make logo`, `make synth`, `make train`, `make blender`, `make eval`, `make backend`, `make plots`), a smoke-test mode (tiny dataset, 1 epoch, 2 trajectories) that runs the whole pipeline end to end in minutes, and CI-style tests.
- Data card and conventions document so the paper can state exactly how data and labels were produced.

## 5. Assumptions and open questions (flag in README, do not block)

- Inverse depth is defined as inverse range (1 / ||p||). If inverse of z-depth is preferred, change in `geometry.py` and in the loss decode only.
- ResNet-54 in the request is treated as ResNet-50.
- Student-t is not natively in GTSAM mEstimators; the spec requires the custom loss and weight route described in M6.
- Intrinsics, marker size (0.25 m), depth range and view-angle limit are initial defaults and live in one config.
- Training budget, equivalence margin and failure thresholds are set in config and should be reviewed after the first smoke run.

## 6. Definition of done

1. `make smoke` completes end to end and produces every figure with tiny data.
2. All unit tests pass (geometry round trip, label reprojection, loss vs scipy, Jacobians, custom robust loss vs head NLL, gating, metrics).
3. Full experiment matrix can be launched with one command, is resumable, and writes the single tidy results file.
4. All figures and tables in M8 are generated from that file.
5. README explains each stage, expected compute and runtime, and how to add a real-data sequence.
6. `docs/results_notes.md` lists supported and unsupported claims with the supporting figures.
