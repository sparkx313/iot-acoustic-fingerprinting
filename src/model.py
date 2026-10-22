"""
IoT Acoustic Fingerprinting — ResNet-18 Mel Spectrogram Classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple

# Device classes (expand this list for real deployments)
DEVICE_CLASSES = [
    "SmartSpeaker_Alexa",
    "SmartSpeaker_Google",
    "SecurityCamera",
    "SmartBulb_Hue",
    "SmartPlug",
    "WiFiRouter",
    "SmartTV",
    "Thermostat",
    "VideoDoorbell",
    "ZigbeeHub",
    "Laptop_Fan",
    "Desktop_PC",
    "AirConditioner",
    "Refrigerator",
    "Unknown",
]

NUM_CLASSES = len(DEVICE_CLASSES)


class AcousticFingerprintNet(nn.Module):
    """
    Modified ResNet-18 for 1-channel mel spectrogram input.
    Input:  (B, 1, n_mels, n_frames) mel spectrogram
    Output: (B, NUM_CLASSES) device logits
    """

    def __init__(self, pretrained: bool = True, n_mels: int = 128, dropout: float = 0.4):
        super().__init__()
        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Replace first conv to accept 1-channel audio spectrogram
        first_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        # Initialize with channel-mean of pretrained weights
        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))

        # Replace FC head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class SpecAugment(nn.Module):
    """
    Spectrogram augmentation: frequency and time masking during training.
    Applied as a preprocessing step before the model.
    """

    def __init__(self, freq_mask_param: int = 20, time_mask_param: int = 40,
                 num_freq_masks: int = 2, num_time_masks: int = 2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, n_frames) — apply masks per-sample."""
        if not self.training:
            return x

        x = x.clone()
        B, C, Freq, Time = x.shape

        for b in range(B):
            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = torch.randint(0, self.freq_mask_param, (1,)).item()
                f0 = torch.randint(0, max(Freq - f, 1), (1,)).item()
                x[b, :, f0: f0 + f, :] = 0.0

            # Time masking
            for _ in range(self.num_time_masks):
                t = torch.randint(0, self.time_mask_param, (1,)).item()
                t0 = torch.randint(0, max(Time - t, 1), (1,)).item()
                x[b, :, :, t0: t0 + t] = 0.0

        return x


def compute_confidence(logits: torch.Tensor) -> Tuple[str, float]:
    """Return predicted class name and softmax confidence."""
    probs = F.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    return DEVICE_CLASSES[idx.item()], float(conf.item())
