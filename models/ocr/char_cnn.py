#!/usr/bin/env python3
from __future__ import annotations
import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Lightweight CNN for character classification (A-Z, 0-9)."""

    def __init__(self, num_classes: int = 36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # robust spatial pooling
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_charcnn(model_path: str, classes_path: str) -> tuple[CharCNN, List[str]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(classes_path, 'r') as f:
        classes: List[str] = json.load(f)
    model = CharCNN(num_classes=len(classes))
    state = torch.load(model_path, map_location=device)
    # Support both plain and 'state_dict' wrapped checkpoints
    state_dict = state.get('state_dict', state)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, classes



