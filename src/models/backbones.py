"""
Backbone architectures adapted to output stride-8 feature maps.
ResNet-9 (DAWNBench style), ResNet-18, and ResNet-50.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv2(self.conv1(x)))


class ResNet9(nn.Module):
    """
    DAWNBench style small ResNet adapted to produce stride-8 features.
    Total stride: Prep (stride 1) -> Layer 1 (stride 2) -> Layer 2 (stride 2) -> Layer 3 (stride 2) = stride 8.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.prep = ConvBNReLU(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        # Layer 1: stride 2 -> total stride 2
        self.layer1_conv = ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer1_res = ResBlock(128)
        
        # Layer 2: stride 2 -> total stride 4
        self.layer2_conv = ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer2_res = ResBlock(256)
        
        # Layer 3: stride 2 -> total stride 8
        self.layer3_conv = ConvBNReLU(256, out_channels, kernel_size=3, stride=2, padding=1)
        self.layer3_res = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 480, 640]
        x = self.prep(x)
        x = self.layer1_res(self.layer1_conv(x))  # [B, 128, 240, 320]
        x = self.layer2_res(self.layer2_conv(x))  # [B, 256, 120, 160]
        x = self.layer3_res(self.layer3_conv(x))  # [B, 256, 60, 80]
        return x


class ResNet18Stride8(nn.Module):
    """
    Standard torchvision ResNet-18 adapted for stride 8 output.
    Initial conv+pool gives stride 4.
    layer1: stride 1 (accum stride 4)
    layer2: stride 2 (accum stride 8)
    layer3 & layer4: dilated / strided to maintain stride 8 and merged via FPN neck.
    """
    def __init__(self, in_channels: int = 3, neck_channels: int = 128):
        super().__init__()
        import torchvision.models as models
        base = models.resnet18(weights=None)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool  # Stride 4
        
        self.layer1 = base.layer1    # Stride 4, 64 channels
        self.layer2 = base.layer2    # Stride 8, 128 channels
        
        # Lateral projection to neck_channels
        self.lat2 = nn.Conv2d(128, neck_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)  # [B, 128, H/8, W/8]
        return self.lat2(x2)


class ResNet50Stride8(nn.Module):
    """
    Standard torchvision ResNet-50 adapted for stride 8 output.
    Uses layer1 (stride 4) and layer2 (stride 8) with 1x1 lateral projection.
    """
    def __init__(self, in_channels: int = 3, neck_channels: int = 128):
        super().__init__()
        import torchvision.models as models
        base = models.resnet50(weights=None)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool  # Stride 4
        
        self.layer1 = base.layer1    # Stride 4, 256 channels
        self.layer2 = base.layer2    # Stride 8, 512 channels
        
        self.lat2 = nn.Conv2d(512, neck_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)  # [B, 512, H/8, W/8]
        return self.lat2(x2)

class DepthwiseConvBNReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pw_bn(self.pw(self.relu(self.dw_bn(self.dw(x))))))


class DepthwiseResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseConvBNReLU(channels, channels, 3, 1, 1)
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.dw_bn(self.dw(out))
        out = self.pw_bn(self.pw(out))
        return self.relu(x + out)


class DepthwiseResNet9Slim(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.prep = DepthwiseConvBNReLU(in_channels, 32, 3, 1, 1)
        self.layer1_conv = DepthwiseConvBNReLU(32, 64, 3, 2, 1)
        self.layer1_res = DepthwiseResBlock(64)
        self.layer2_conv = DepthwiseConvBNReLU(64, 128, 3, 2, 1)
        self.layer2_res = DepthwiseResBlock(128)
        self.layer3_conv = DepthwiseConvBNReLU(128, out_channels, 3, 2, 1)
        self.layer3_res = DepthwiseResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1_res(self.layer1_conv(x))
        x = self.layer2_res(self.layer2_conv(x))
        x = self.layer3_res(self.layer3_conv(x))
        return x


class DepthwiseResNet6(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.prep = DepthwiseConvBNReLU(in_channels, 32, 3, 2, 1)
        self.layer1_conv = DepthwiseConvBNReLU(32, 64, 3, 2, 1)
        self.layer2_conv = DepthwiseConvBNReLU(64, out_channels, 3, 2, 1)
        self.layer2_res = DepthwiseResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1_conv(x)
        x = self.layer2_res(self.layer2_conv(x))
        return x


def build_backbone(backbone_type: str = "resnet18", neck_channels: int = 128) -> nn.Module:
    """Builds backbone adapted to output stride-8 feature map with neck_channels."""
    name = backbone_type.lower()
    if name in ["resnet9", "resnet-9"]:
        return nn.Sequential(
            ResNet9(in_channels=3, out_channels=256),
            nn.Conv2d(256, neck_channels, 1)
        )
    elif name in ["depthresnet9", "depthresnet-9"]:
        return nn.Sequential(
            DepthwiseResNet9Slim(in_channels=3, out_channels=256),
            nn.Conv2d(256, neck_channels, 1)
        )
    elif name in ["depthresnet6", "depthresnet-6"]:
        return nn.Sequential(
            DepthwiseResNet6(in_channels=3, out_channels=256),
            nn.Conv2d(256, neck_channels, 1)
        )
    elif name in ["resnet18", "resnet-18"]:
        return ResNet18Stride8(in_channels=3, neck_channels=neck_channels)
    elif name in ["resnet50", "resnet-50", "resnet54", "resnet-54"]:
        return ResNet50Stride8(in_channels=3, neck_channels=neck_channels)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
