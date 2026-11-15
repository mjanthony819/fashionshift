"""Model definitions shared by training, evaluation and export.

The JS re-implementation in js/inference.js mirrors SmallCNN exactly;
tests/test_parity.py proves the two agree.
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """conv(1->16,3x3)-relu-maxpool2 -> conv(16->32,3x3)-relu-maxpool2 -> fc(1568->128)-relu -> fc(128->10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (N, 784) or (N, 1, 28, 28), values already standardized
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(torch.relu(self.conv2(x)))   # 14 -> 7
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                          # logits


def param_count(model):
    return sum(p.numel() for p in model.parameters())
