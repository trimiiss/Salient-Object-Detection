"""
sod_model.py
============
CNN Encoder-Decoder for Salient Object Detection — built from scratch.
Two variants:
  • SODNet       — baseline encoder-decoder (no BN, no skip connections)
  • SODNetPlus   — improved version with BatchNorm, Dropout, skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    """Conv2d → (optional) BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, use_bn=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpConvBnRelu(nn.Module):
    """ConvTranspose2d → (optional) BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, use_bn=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# BASELINE MODEL  (SODNet)
# ─────────────────────────────────────────────
class SODNet(nn.Module):
    """
    Simple encoder-decoder CNN.
    Input : (B, 3, 224, 224)
    Output: (B, 1, 224, 224)  — sigmoid saliency mask
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────
        self.enc1 = ConvBnRelu(3,   32, use_bn=False)   # → (B,32,224,224)
        self.pool1 = nn.MaxPool2d(2)                     # → (B,32,112,112)

        self.enc2 = ConvBnRelu(32,  64, use_bn=False)   # → (B,64,112,112)
        self.pool2 = nn.MaxPool2d(2)                     # → (B,64, 56, 56)

        self.enc3 = ConvBnRelu(64, 128, use_bn=False)   # → (B,128, 56, 56)
        self.pool3 = nn.MaxPool2d(2)                     # → (B,128, 28, 28)

        self.enc4 = ConvBnRelu(128, 256, use_bn=False)  # → (B,256, 28, 28)
        self.pool4 = nn.MaxPool2d(2)                     # → (B,256, 14, 14)

        # ── Bottleneck ───────────────────────
        self.bottleneck = ConvBnRelu(256, 512, use_bn=False)  # → (B,512,14,14)

        # ── Decoder ──────────────────────────
        self.up4  = UpConvBnRelu(512, 256, use_bn=False)  # → (B,256,28,28)
        self.dec4 = ConvBnRelu(256, 256, use_bn=False)

        self.up3  = UpConvBnRelu(256, 128, use_bn=False)  # → (B,128,56,56)
        self.dec3 = ConvBnRelu(128, 128, use_bn=False)

        self.up2  = UpConvBnRelu(128,  64, use_bn=False)  # → (B,64,112,112)
        self.dec2 = ConvBnRelu(64,  64, use_bn=False)

        self.up1  = UpConvBnRelu(64,   32, use_bn=False)  # → (B,32,224,224)
        self.dec1 = ConvBnRelu(32,  32, use_bn=False)

        # ── Output head ──────────────────────
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x);  p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder (no skip connections in baseline)
        d4 = self.dec4(self.up4(b))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        return torch.sigmoid(self.out_conv(d1))


# ─────────────────────────────────────────────
# IMPROVED MODEL  (SODNetPlus)
# ─────────────────────────────────────────────
class SODNetPlus(nn.Module):
    """
    Enhanced encoder-decoder with:
      • BatchNorm after every convolution
      • Skip connections (U-Net style)
      • Dropout in bottleneck
    Input : (B, 3, 224, 224)
    Output: (B, 1, 224, 224)
    """

    def __init__(self, dropout_p=0.4):
        super().__init__()

        # ── Encoder (with BN) ─────────────────
        self.enc1 = ConvBnRelu(3,   32, use_bn=True)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBnRelu(32,  64, use_bn=True)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBnRelu(64, 128, use_bn=True)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBnRelu(128, 256, use_bn=True)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck + Dropout ──────────────
        self.bottleneck = nn.Sequential(
            ConvBnRelu(256, 512, use_bn=True),
            nn.Dropout2d(p=dropout_p),
            ConvBnRelu(512, 512, use_bn=True),
        )

        # ── Decoder (skip connections) ────────
        # After concat: channels doubled
        self.up4  = UpConvBnRelu(512, 256, use_bn=True)
        self.dec4 = ConvBnRelu(256+256, 256, use_bn=True)   # skip from enc4

        self.up3  = UpConvBnRelu(256, 128, use_bn=True)
        self.dec3 = ConvBnRelu(128+128, 128, use_bn=True)   # skip from enc3

        self.up2  = UpConvBnRelu(128,  64, use_bn=True)
        self.dec2 = ConvBnRelu(64+64,   64, use_bn=True)    # skip from enc2

        self.up1  = UpConvBnRelu(64,   32, use_bn=True)
        self.dec1 = ConvBnRelu(32+32,   32, use_bn=True)    # skip from enc1

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x);  p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b),   e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2),  e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────
def iou_loss(pred, target, smooth=1e-6):
    """Differentiable IoU loss."""
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union        = pred_flat.sum() + target_flat.sum() - intersection
    iou          = (intersection + smooth) / (union + smooth)
    return 1.0 - iou


def combined_loss(pred, target, bce_weight=1.0, iou_weight=0.5):
    """BCE + weighted IoU loss as required by the spec."""
    bce = F.binary_cross_entropy(pred, target)
    iou = iou_loss(pred, target)
    return bce_weight * bce + iou_weight * iou


# ─────────────────────────────────────────────
# Model summary helper
# ─────────────────────────────────────────────
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    return trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("── Baseline SODNet ──")
    model = SODNet().to(device)
    count_parameters(model)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Output shape: {out.shape}\n")   # expect (2,1,224,224)

    print("── Improved SODNetPlus ──")
    model2 = SODNetPlus().to(device)
    count_parameters(model2)
    out2 = model2(x)
    print(f"Output shape: {out2.shape}")
