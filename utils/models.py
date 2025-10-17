import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import norm
import time
import gc
from pathlib import Path
from itertools import cycle
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set device to CPU as requested
device = torch.device("cpu")


@dataclass
class ModelSpec:
    title: str
    checkpoint_path: str
    num_classes: int
    task_type: str  # 'multiclass' or 'binary'
    class_names: list = None  # Class names for this model


# Helper classes for TriadAttention
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(ConvBlock, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = ConvBlock(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TriadAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TriadAttention, self).__init__()
        self.channel_wise = AttentionGate()
        self.height_wise = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.spatial_wise = AttentionGate()

    def forward(self, x):
        logger.debug(f"TriadAttention input shape: {x.shape}")
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.channel_wise(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.height_wise(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.spatial_wise(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        logger.debug(f"TriadAttention output shape: {x.shape}")
        return x_out


# Helper classes for ConvNeXtBlock
class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        logger.debug(f"LayerNorm2d input shape: {x.shape}")
        x = x.contiguous()
        x = x.reshape(-1, channels)
        x = self.ln(x)
        x = x.reshape(batch, height, width, channels)
        logger.debug(f"LayerNorm2d output shape: {x.shape}")
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        if self.mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
            noise = x.new_tensor(data=torch.ones(shape)).uniform_(0.0, 1.0) < self.p
            return x.div_(1.0 - self.p).mul_(noise)
        elif self.mode == "batch":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
            noise = x.new_tensor(data=torch.ones(shape)).uniform_(0.0, 1.0) < self.p
            return x.div_(1.0 - self.p).mul_(noise)
        else:
            raise NotImplementedError


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            LayerNorm2d(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logger.debug(f"ConvNeXtBlock input shape: {input.shape}")
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        logger.debug(f"ConvNeXtBlock output shape: {result.shape}")
        return result


# Channel Attention (CBAM-based)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logger.debug(f"ChannelAttention input shape: {x.shape}")
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        out = self.sigmoid(avg_out + max_out)
        x = x * out
        logger.debug(f"ChannelAttention output shape: {x.shape}")
        return x


# Spatial Attention (CBAM-based)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logger.debug(f"SpatialAttention input shape: {x.shape}")
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        x = x * out
        logger.debug(f"SpatialAttention output shape: {x.shape}")
        return x


# EnhancedBlock (simplified to one ConvNeXtBlock)
class EnhancedBlock(nn.Module):
    def __init__(self, dim, layer_scale=1e-6, stochastic_depth_prob=0.0):
        super().__init__()
        self.convnext = ConvNeXtBlock(dim, layer_scale, stochastic_depth_prob)

    def forward(self, x):
        logger.debug(f"EnhancedBlock input shape: {x.shape}")
        x = self.convnext(x)
        logger.debug(f"EnhancedBlock output shape: {x.shape}")
        return x


# Custom model - EnhancedMobileNetV2
class EnhancedMobileNetV2(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        features_list = list(base_model.features.children())

        # Insert TriadAttention after specified inverted residual blocks
        self.features = nn.Sequential(
            *features_list[0:7],  # Up to block 6
            TriadAttention(),
            *features_list[7:11],  # Blocks 7-10
            TriadAttention(),
            *features_list[11:14],  # Blocks 11-13
            TriadAttention(),
            *features_list[14:],  # Remaining blocks
        )

        # Add the enhanced blocks with ChannelAttention and SpatialAttention
        self.enhanced_blocks = nn.Sequential(
            EnhancedBlock(1280),
            ChannelAttention(1280),
            EnhancedBlock(1280),
            SpatialAttention(),
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),  # MobileNetV2 outputs 1280 channels
        )

    def forward(self, x):
        logger.debug(f"Model input shape: {x.shape}")
        x = self.features(x)
        logger.debug(f"After features shape: {x.shape}")
        x = self.enhanced_blocks[0](x)  # First EnhancedBlock
        logger.debug(f"After enhanced_blocks[0] shape: {x.shape}")
        x = self.enhanced_blocks[1](x)  # ChannelAttention
        logger.debug(f"After enhanced_blocks[1] (ChannelAttention) shape: {x.shape}")
        x = self.enhanced_blocks[2](x)  # Second EnhancedBlock
        logger.debug(f"After enhanced_blocks[2] shape: {x.shape}")
        x = self.enhanced_blocks[3](x)  # SpatialAttention
        logger.debug(f"After enhanced_blocks[3] (SpatialAttention) shape: {x.shape}")
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        logger.debug(f"Model output shape: {x.shape}")
        return x


# Resolve checkpoint absolute path regardless of where Streamlit is launched
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))


def _abs_ckpt(rel_path: str) -> str:
    candidates = [
        os.path.join(WORKSPACE_ROOT, rel_path),  # sibling to app dir
        os.path.join(APP_DIR, rel_path),  # under app dir (fallback)
        os.path.abspath(rel_path),  # as-given absolute/working dir
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # If not found, return first candidate to show useful error path
    return candidates[0]


# Model registry with direct checkpoint paths
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "8-Class Classification": ModelSpec(
        title="8-Class Classification",
        checkpoint_path=_abs_ckpt("Best_Models/8_cls/best_model_fold_3.pth"),
        num_classes=8,
        task_type="multiclass",
        class_names=[
            "normal-cecum",
            "dyed-lifted-polyps",
            "dyed-resection-margins",
            "esophagitis",
            "normal-pylorus",
            "polyps",
            "ulcerative-colitis",
            "normal-z-line",
        ],
    ),
    "6-Class Classification": ModelSpec(
        title="6-Class Classification",
        checkpoint_path=_abs_ckpt("Best_Models/6_cls/best_model_fold_3.pth"),
        num_classes=6,
        task_type="multiclass",
        class_names=[
            "normal-cecum",
            "esophagitis",
            "normal-pylorus",
            "polyps",
            "ulcerative-colitis",
            "normal-z-line",
        ],
    ),
    # "Anatomical Classification": ModelSpec(
    #     title="Anatomical Classification",
    #     checkpoint_path=_abs_ckpt(
    #         "MNv2_SA_Anatomical/Best_States/best_model_fold_1.pth"
    #     ),
    #     num_classes=3,
    #     task_type="multiclass",
    #     class_names=["normal-cecum", "normal-pylorus", "normal-z-line"],
    # ),
    # "Pathological Classification": ModelSpec(
    #     title="Pathological Classification",
    #     checkpoint_path=_abs_ckpt("MNv2_SA_4_cls/Best_States/best_model_fold_3.pth"),
    #     num_classes=4,
    #     task_type="multiclass",
    #     class_names=["Normal", "Esophagitis", "Polyps", "Ulcerative-Colitis"],
    # ),
    "Normal vs. Abnormal Classification": ModelSpec(
        title="Normal vs. Abnormal Classification",
        checkpoint_path=_abs_ckpt("Best_Models/NAb/best_model_fold_3.pth"),
        num_classes=1,
        task_type="binary",
        class_names=["Normal", "Abnormal"],
    ),
    "Normal vs. Distinct Diseases": ModelSpec(
        title="Normal vs. Distinct Diseases",
        checkpoint_path=_abs_ckpt("Best_Models/NDi/best_model_fold_3.pth"),
        num_classes=4,
        task_type="multiclass",
        class_names=["Normal", "Esophagitis", "Polyps", "Ulcerative-Colitis"],
    ),
    "Polyps vs. Non-Polyps": ModelSpec(
        title="Polyps vs. Non-Polyps",
        checkpoint_path=_abs_ckpt("Best_Models/PnP/best_model_fold_1.pth"),
        num_classes=1,
        task_type="binary",
        class_names=["Non-Polyps", "Polyps"],
    ),
}


def get_model(num_classes=6):
    """Initialize EnhancedMobileNetV2 with specified number of classes"""
    logger.info(f"Initializing EnhancedMobileNetV2 with {num_classes} classes")
    model = EnhancedMobileNetV2(num_classes=num_classes)
    return model.to(device)


def load_model(scheme: str) -> Tuple[torch.nn.Module, torch.device, ModelSpec]:
    """Load model for the specified classification scheme"""
    if scheme not in MODEL_REGISTRY:
        raise ValueError(f"Unknown scheme: {scheme}")

    spec = MODEL_REGISTRY[scheme]

    # Initialize model with correct number of classes
    model = get_model(num_classes=spec.num_classes)

    # Load checkpoint
    checkpoint_path = spec.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device, spec


def predict_image(
    model: torch.nn.Module,
    device: torch.device,
    image_tensor: torch.Tensor,
    task_type: str,
    class_names: list = None,
) -> Tuple[int, torch.Tensor, str]:
    """Predict class for an image tensor"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        if task_type == "binary":
            probs = torch.sigmoid(outputs)
            pred = int((probs > 0.5).item())
            class_name = class_names[pred] if class_names else str(pred)
            return pred, probs.squeeze().detach().cpu(), class_name
        probs = torch.softmax(outputs, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        class_name = class_names[pred] if class_names else str(pred)
        return pred, probs.squeeze().detach().cpu(), class_name
