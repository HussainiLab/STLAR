import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1D ResNet for HFO detection."""
    def __init__(self, n_channels=1, num_classes=1):
        super().__init__()
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            in_channels = bottleneck_channels
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2, bias=False) 
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        
        outs = [conv(x) for conv in self.convs]
        outs.append(self.conv_pool(self.maxpool(input_tensor if not self.use_bottleneck else x)))
        
        x = torch.cat(outs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionTime(nn.Module):
    def __init__(self, n_channels=1, num_classes=1, num_modules=6):
        super().__init__()
        self.num_modules = num_modules
        self.out_channels = 32
        
        self.modules_list = nn.ModuleList()
        for i in range(num_modules):
            self.modules_list.append(InceptionModule(
                in_channels=n_channels if i == 0 else self.out_channels * 4, 
                out_channels=self.out_channels
            ))
            
        self.shortcuts = nn.ModuleList()
        for i in range(num_modules // 3):
            c_in = n_channels if i == 0 else self.out_channels * 4
            c_out = self.out_channels * 4
            if c_in != c_out:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
                    nn.BatchNorm1d(c_out)
                ))
            else:
                self.shortcuts.append(nn.Identity())
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.out_channels * 4, num_classes)

    def forward(self, x):
        res = x
        for i, mod in enumerate(self.modules_list):
            x = mod(x)
            if (i + 1) % 3 == 0:
                shortcut_idx = (i + 1) // 3 - 1
                s = self.shortcuts[shortcut_idx](res)
                x = x + s
                x = F.relu(x)
                res = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class HFOTransformer(nn.Module):
    """Simple 1D Transformer for time-series classification."""
    def __init__(self, n_channels=1, num_classes=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # Project input to d_model dimensions
        self.embedding = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=10, stride=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        # Learnable positional encoding (max len 1000 after stride)
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, 1000) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) # (B, d_model, L)
        
        # Add positional encoding (truncate to current length)
        L = x.shape[2]
        if L <= self.pos_encoder.shape[2]:
            x = x + self.pos_encoder[:, :, :L]
        
        x = x.permute(0, 2, 1) # (B, L, d_model) for Transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) # Global Average Pooling
        x = self.fc(x)
        return x


class Spectrogram2DCNN(nn.Module):
    """Computes STFT on-the-fly and applies 2D CNN."""
    def __init__(self, n_channels=1, num_classes=1):
        super().__init__()
        self.n_fft = 128
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, L) -> STFT -> (B, F, T)
        # return_complex=True is default in newer torch, we need magnitude
        spec = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.n_fft//4, 
                          window=torch.hann_window(self.n_fft).to(x.device), return_complex=True)
        img = torch.log1p(torch.abs(spec)).unsqueeze(1) # (B, 1, F, T)
        
        feat = self.net(img).flatten(1)
        return self.fc(feat)


def build_model(model_type=2):
    """Factory function to build model based on ID."""
    if model_type == 1:
        return Simple1DCNN(n_channels=1)
    elif model_type == 2:
        return ResNet1D(n_channels=1)
    elif model_type == 3:
        return InceptionTime(n_channels=1)
    elif model_type == 4:
        return HFOTransformer(n_channels=1)
    elif model_type == 5:
        return Spectrogram2DCNN(n_channels=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Options: 1=SimpleCNN, 2=ResNet1D, 3=InceptionTime, 4=Transformer, 5=2D_CNN")
