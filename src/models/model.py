"""
Complete Marker Localization Model uniting Backbone and CenterNet Uncertainty Head.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from src.models.backbones import build_backbone
from src.models.heads import CenterNetUncertaintyHead
from src.models.decode import decode_detections
from src.common.geometry import DEFAULT_INTRINSICS, CameraIntrinsics


class MarkerModel(nn.Module):
    def __init__(
        self,
        backbone_type: str = "resnet18",
        head_type: str = "gaussian",
        neck_channels: int = 128,
        head_channels: int = 128,
        fixed_nu: float = 3.0,
        log_scale_min: float = -7.0,
        log_scale_max: float = 5.0,
        intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS,
        stride: int = 8,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.head_type = head_type
        self.stride = stride
        self.intrinsics = intrinsics

        # Backbone produces stride-8 feature map
        self.backbone = build_backbone(backbone_type, neck_channels=neck_channels)

        # CenterNet dense uncertainty head
        self.head = CenterNetUncertaintyHead(
            in_channels=neck_channels,
            head_channels=head_channels,
            head_type=head_type,
            fixed_nu=fixed_nu,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            intrinsics=intrinsics,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return self.head(feat)

    def freeze_uncertainty_branch(self) -> None:
        self.head.freeze_uncertainty_branch()

    def unfreeze_uncertainty_branch(self) -> None:
        self.head.unfreeze_uncertainty_branch()

    def initialize_uncertainty_bias(self, residual_rms: torch.Tensor) -> None:
        self.head.initialize_uncertainty_bias(residual_rms)

    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.3,
        max_detections: int = 20,
    ):
        """Inference helper performing forward pass, NMS, and measurement decoding."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return decode_detections(
                outputs,
                score_threshold=score_threshold,
                max_detections=max_detections,
                stride=self.stride,
                intrinsics=self.intrinsics,
            )
