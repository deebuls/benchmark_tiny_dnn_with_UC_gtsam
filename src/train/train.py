"""
Two-Stage Trainer for Marker Localization Network with Uncertainty Heads.
Stage 1: Mean-only warm up (robust smooth L1, uncertainty head frozen).
Stage 2: Joint training (NLL heads active, initialized from stage 1 residual RMS).
Includes EMA of weights, mixed precision, gradient clipping, and logging.
"""

import os
import copy
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.models.model import MarkerModel
from src.data.dataset import MarkerDataset
from src.losses.uncertainty_losses import (
    focal_heatmap_loss,
    smooth_l1_regression_loss,
    gaussian_nll_loss,
    gaussian_beta_nll_loss,
    student_t_nll_loss,
    cauchy_nll_loss,
)


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for p_ema, p in zip(self.ema.parameters(), model.parameters()):
                p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


def compute_stage1_residual_rms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> torch.Tensor:
    """
    Computes RMS of residuals on stage-1 outputs to initialize uncertainty bias for Stage 2.
    Residuals for: [delta_azimuth, delta_elevation, log_rho].
    """
    model.eval()
    sq_errs = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx >= max_batches:
                break
            imgs = batch["image"].to(device)
            reg_target = batch["reg_target"].to(device)
            reg_mask = batch["reg_mask"].to(device)

            preds = model(imgs)
            mu = preds["mu"]

            # Compute squared error where mask is 1
            diff_sq = (mu - reg_target) ** 2  # [B, 3, H, W]
            mask_exp = reg_mask.expand_as(diff_sq)
            valid_errs = diff_sq[mask_exp > 0].view(-1, 3)
            if valid_errs.shape[0] > 0:
                sq_errs.append(valid_errs)

    if len(sq_errs) == 0:
        return torch.tensor([0.1, 0.1, 0.2], device=device)

    all_sq = torch.cat(sq_errs, dim=0)
    rms = torch.sqrt(torch.mean(all_sq, dim=0))
    return rms


def evaluate_validation_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    head_type: str,
) -> Dict[str, float]:
    """Runs a validation pass computing validation loss, MAE, and RMS errors."""
    model.eval()
    total_loss, total_focal, total_nll = 0.0, 0.0, 0.0
    abs_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            heatmap_gt = batch["heatmap"].to(device)
            ignore_mask = batch["ignore_mask"].to(device)
            reg_mask = batch["reg_mask"].to(device)
            reg_target = batch["reg_target"].to(device)

            preds = model(imgs)
            l_focal = focal_heatmap_loss(preds["obj_logits"], heatmap_gt, ignore_mask)

            mu = preds["mu"]
            log_scale = preds.get("log_scale")
            if head_type == "gaussian":
                l_nll = gaussian_nll_loss(mu, log_scale, reg_target, reg_mask)
            elif head_type == "gaussian_beta":
                l_nll = gaussian_beta_nll_loss(mu, log_scale, reg_target, reg_mask)
            elif head_type in ["student_t", "student_t_fixed"]:
                nu = preds.get("nu", torch.full_like(mu, 3.0))
                l_nll = student_t_nll_loss(mu, log_scale, nu, reg_target, reg_mask)
            elif head_type == "cauchy":
                l_nll = cauchy_nll_loss(mu, log_scale, reg_target, reg_mask)
            else:
                l_nll = smooth_l1_regression_loss(mu, reg_target, reg_mask)

            loss = l_focal + l_nll
            total_loss += loss.item()
            total_focal += l_focal.item()
            total_nll += l_nll.item()

            # Measure point residuals where targets exist
            mask_exp = reg_mask.expand_as(mu)
            if mask_exp.sum() > 0:
                diff = torch.abs(mu - reg_target)
                valid_diff = diff[mask_exp > 0].view(-1, 3)
                abs_errors.append(valid_diff.cpu().numpy())

    n_b = max(1, len(val_loader))
    val_metrics = {
        "val_total_loss": total_loss / n_b,
        "val_focal_loss": total_focal / n_b,
        "val_nll_loss": total_nll / n_b,
    }

    if abs_errors:
        all_diffs = np.concatenate(abs_errors, axis=0)
        maes = np.mean(all_diffs, axis=0)
        rmses = np.sqrt(np.mean(all_diffs**2, axis=0))
        val_metrics["val_az_mae_deg"] = float(np.rad2deg(maes[0]))
        val_metrics["val_az_rmse_deg"] = float(np.rad2deg(rmses[0]))
        val_metrics["val_el_mae_deg"] = float(np.rad2deg(maes[1]))
        val_metrics["val_el_rmse_deg"] = float(np.rad2deg(rmses[1]))
        val_metrics["val_rho_mae"] = float(maes[2])
        val_metrics["val_rho_rmse"] = float(rmses[2])

    return val_metrics


def train_model(
    config: Dict[str, Any],
    train_dir: str,
    val_dir: str,
    output_dir: str,
) -> None:
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    print(f"Training on device: {device}")

    # Datasets and Loaders
    train_ds = MarkerDataset(train_dir, is_train=True)
    val_ds = MarkerDataset(val_dir, is_train=False)

    batch_sz = config.get("batch_size", 8)
    workers = config.get("num_workers", 2)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False, num_workers=workers)

    # Instantiate model
    backbone_type = config["backbone"]["type"]
    head_type = config["head"]["type"]
    model = MarkerModel(
        backbone_type=backbone_type,
        head_type=head_type,
        neck_channels=config["backbone"].get("neck_channels", 128),
        head_channels=config["backbone"].get("head_channels", 128),
        fixed_nu=config["head"].get("fixed_nu", 3.0),
        log_scale_min=config["head"].get("log_scale_min", -7.0),
        log_scale_max=config["head"].get("log_scale_max", 5.0),
    ).to(device)

    ema = ModelEMA(model, decay=config.get("ema", {}).get("decay", 0.999))
    use_amp = config.get("mixed_precision", True) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_epochs = config.get("total_epochs", 30)
    stage1_frac = config.get("stage1_fraction", 0.2)
    stage1_epochs = max(1, int(round(total_epochs * stage1_frac)))
    stage2_epochs = total_epochs - stage1_epochs

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Initialize training history log
    training_history = []

    # ========================================================
    # STAGE 1: Mean-only warm up (Uncertainty head frozen)
    # ========================================================
    print(f"\n=== Starting Stage 1 Warm-up: {stage1_epochs} epochs ===")
    model.freeze_uncertainty_branch()
    if config.get("freeze_backbone_stage1", False):
        for p in model.backbone.parameters():
            p.requires_grad = False

    opt_stage1 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["optimizer"]["lr_stage1"]["backbone"]},
        {"params": model.head.obj_branch.parameters(), "lr": config["optimizer"]["lr_stage1"]["neck_head"]},
        {"params": model.head.mean_branch.parameters(), "lr": config["optimizer"]["lr_stage1"]["neck_head"]},
    ], weight_decay=config["optimizer"].get("weight_decay", 0.01))

    for epoch in range(1, stage1_epochs + 1):
        model.train()
        total_loss, total_focal, total_reg = 0.0, 0.0, 0.0
        opt_stage1.zero_grad()
        for step_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            heatmap_gt = batch["heatmap"].to(device)
            ignore_mask = batch["ignore_mask"].to(device)
            reg_mask = batch["reg_mask"].to(device)
            reg_target = batch["reg_target"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(imgs)
                l_focal = focal_heatmap_loss(preds["obj_logits"], heatmap_gt, ignore_mask)
                l_reg = smooth_l1_regression_loss(preds["mu"], reg_target, reg_mask)
                loss = (l_focal + l_reg) / grad_accum_steps

            scaler.scale(loss).backward()
            
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(opt_stage1)
                nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"].get("clip_grad_norm", 5.0))
                scaler.step(opt_stage1)
                scaler.update()
                opt_stage1.zero_grad()
                ema.update(model)

            total_loss += loss.item() * grad_accum_steps
            total_focal += l_focal.item()
            total_reg += l_reg.item()

        n_b = max(1, len(train_loader))
        train_loss_epoch = total_loss / n_b
        print(f"Stage 1 Epoch {epoch}/{stage1_epochs} - Loss: {train_loss_epoch:.4f} (Focal: {total_focal/n_b:.4f}, Reg: {total_reg/n_b:.4f})")

        # Record Stage 1 epoch log
        training_history.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": float(train_loss_epoch),
            "train_focal": float(total_focal / n_b),
            "train_reg_or_nll": float(total_reg / n_b),
        })

    # Save Stage 1 Checkpoint
    torch.save(model.state_dict(), str(out_path / "checkpoint_stage1.pt"))

    # ========================================================
    # STAGE 2: Joint Training with Uncertainty Head Active
    # ========================================================
    print(f"\n=== Initializing Stage 2 Joint Training: {stage2_epochs} epochs ===")
    # Unfreeze uncertainty branch and backbone
    model.unfreeze_uncertainty_branch()
    for p in model.backbone.parameters():
        p.requires_grad = True

    # Initialize uncertainty branch bias from Stage 1 residual RMS
    res_rms = compute_stage1_residual_rms(model, train_loader, device)
    model.initialize_uncertainty_bias(res_rms)

    # Stage 2 Optimizer with differential learning rates
    opt_params = [
        {"params": model.backbone.parameters(), "lr": config["optimizer"]["lr_stage2"]["backbone"]},
        {"params": model.head.obj_branch.parameters(), "lr": config["optimizer"]["lr_stage2"]["neck_head"]},
        {"params": model.head.mean_branch.parameters(), "lr": config["optimizer"]["lr_stage2"]["neck_head"]},
    ]
    if model.head.unc_branch is not None:
        opt_params.append({"params": model.head.unc_branch.parameters(), "lr": config["optimizer"]["lr_stage2"]["uncertainty"]})
    if model.head.nu_branch is not None:
        opt_params.append({"params": model.head.nu_branch.parameters(), "lr": config["optimizer"]["lr_stage2"]["uncertainty"]})

    opt_stage2 = torch.optim.AdamW(opt_params, weight_decay=config["optimizer"].get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_stage2, T_max=max(1, stage2_epochs))

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        total_loss, total_focal, total_nll = 0.0, 0.0, 0.0
        opt_stage2.zero_grad()
        for step_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            heatmap_gt = batch["heatmap"].to(device)
            ignore_mask = batch["ignore_mask"].to(device)
            reg_mask = batch["reg_mask"].to(device)
            reg_target = batch["reg_target"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(imgs)
                l_focal = focal_heatmap_loss(preds["obj_logits"], heatmap_gt, ignore_mask)

                # NLL Loss selection based on head_type
                mu = preds["mu"]
                log_scale = preds.get("log_scale")
                
                if head_type == "gaussian":
                    l_nll = gaussian_nll_loss(mu, log_scale, reg_target, reg_mask)
                elif head_type == "gaussian_beta":
                    beta_val = config["head"].get("beta_nll", 0.5)
                    l_nll = gaussian_beta_nll_loss(mu, log_scale, reg_target, reg_mask, beta=beta_val)
                elif head_type in ["student_t", "student_t_fixed"]:
                    nu = preds["nu"]
                    l_nll = student_t_nll_loss(mu, log_scale, nu, reg_target, reg_mask)
                elif head_type == "cauchy":
                    l_nll = cauchy_nll_loss(mu, log_scale, reg_target, reg_mask)
                elif head_type == "mse_fixed":
                    l_nll = smooth_l1_regression_loss(mu, reg_target, reg_mask)
                else:
                    raise ValueError(f"Unknown head type: {head_type}")

                loss = (l_focal + l_nll) / grad_accum_steps

            scaler.scale(loss).backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(opt_stage2)
                nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"].get("clip_grad_norm", 5.0))
                scaler.step(opt_stage2)
                scaler.update()
                opt_stage2.zero_grad()
                ema.update(model)

            total_loss += loss.item() * grad_accum_steps
            total_focal += l_focal.item()
            total_nll += l_nll.item()

        scheduler.step()
        n_b = max(1, len(train_loader))
        train_loss_epoch = total_loss / n_b
        print(f"Stage 2 Epoch {epoch}/{stage2_epochs} - Loss: {train_loss_epoch:.4f} (Focal: {total_focal/n_b:.4f}, NLL: {total_nll/n_b:.4f})")

        # Record Stage 2 epoch log
        training_history.append({
            "stage": 2,
            "epoch": epoch,
            "total_epoch": stage1_epochs + epoch,
            "train_loss": float(train_loss_epoch),
            "train_focal": float(total_focal / n_b),
            "train_reg_or_nll": float(total_nll / n_b),
        })

    # ========================================================
    # FINAL VALIDATION EVALUATION & SUMMARY
    # ========================================================
    print("\n" + "=" * 60)
    print(">>> Running End-of-Training Validation Metrics Evaluation...")
    val_summary = evaluate_validation_epoch(ema.ema, val_loader, device, head_type)

    print("--- Final Validation Metrics (Model EMA) ---")
    print(f"  Val Loss:       {val_summary.get('val_total_loss', 0.0):.4f} (Focal: {val_summary.get('val_focal_loss', 0.0):.4f}, NLL: {val_summary.get('val_nll_loss', 0.0):.4f})")
    if "val_az_mae_deg" in val_summary:
        print(f"  Azimuth Error:   MAE = {val_summary['val_az_mae_deg']:.2f}° | RMSE = {val_summary['val_az_rmse_deg']:.2f}°")
        print(f"  Elevation Error: MAE = {val_summary['val_el_mae_deg']:.2f}° | RMSE = {val_summary['val_el_rmse_deg']:.2f}°")
        print(f"  Inv-Depth (rho): MAE = {val_summary['val_rho_mae']:.3f} | RMSE = {val_summary['val_rho_rmse']:.3f} (1/m)")
    print("=" * 60)

    # Save final models (standard & EMA), config, and training history log
    torch.save(model.state_dict(), str(out_path / "model_final.pt"))
    torch.save(ema.ema.state_dict(), str(out_path / "model_ema_final.pt"))
    with open(out_path / "config.yaml", "w") as f_cfg:
        yaml.safe_dump(config, f_cfg)

    # Save training history and validation summary to train_log.json
    full_log = {
        "config": config,
        "final_validation_metrics": val_summary,
        "training_history": training_history,
    }
    with open(out_path / "train_log.json", "w") as f_log:
        json.dump(full_log, f_log, indent=2)

    print(f"Training completed. Models, logs, and metrics saved in {out_path}/")
