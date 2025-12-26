import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    """A small 1D CNN for binary classification on waveform segments."""

    def __init__(self, n_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        # x: (N, C, L)
        feat = self.net(x).squeeze(-1)
        logit = self.head(feat)
        return logit


def build_model():
    return Simple1DCNN(n_channels=1)
