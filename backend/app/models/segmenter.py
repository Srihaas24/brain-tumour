import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class nnUNetBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class nnUNet2D(nn.Module):
    """
    Standard 2D nnU-Net architecture with InstanceNorm, LeakyReLU,
    strided convolutions, and transpose convolutions for medical image segmentation.
    """
    def __init__(self, in_channels=3, num_classes=1, mc_dropout=0.2):
        super().__init__()
        self.enc1 = nnUNetBlock(in_channels, 32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.enc2 = nnUNetBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.enc3 = nnUNetBlock(128, 128)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.enc4 = nnUNetBlock(256, 256, dropout_p=mc_dropout)
        self.down4 = nn.Conv2d(256, 320, kernel_size=3, stride=2, padding=1)

        # Bottleneck with Monte Carlo Dropout
        self.bottleneck = nnUNetBlock(320, 320, dropout_p=mc_dropout)

        # Decoder stages
        self.up4 = nn.ConvTranspose2d(320, 256, kernel_size=2, stride=2)
        self.dec4 = nnUNetBlock(512, 256, dropout_p=mc_dropout)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nnUNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nnUNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nnUNetBlock(64, 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


_segmenter_cache = {}

def get_segmenter(device=None, weights_path=None):
    """
    Singleton-style getter for the trained nnU-Net segmentation model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'model' in _segmenter_cache:
        return _segmenter_cache['model'], device

    model = nnUNet2D(in_channels=3, num_classes=1, mc_dropout=0.2)
    
    if weights_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        candidate_paths = [
            os.path.join(base_dir, 'models', 'best_nnunet_model.pt'),
            os.path.join(base_dir, 'ALTHON2.0', 'best_nnunet_model.pt')
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                weights_path = p
                break

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[NeuroVision AI] Loaded nnU-Net weights from: {weights_path}")
    else:
        print("[NeuroVision AI] Warning: No nnU-Net weights found. Initialized with default weights.")

    model = model.to(device)
    model.eval()
    _segmenter_cache['model'] = model
    return model, device
