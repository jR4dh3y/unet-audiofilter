
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downsampling block with max pooling"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)

        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """U-Net for Speech Enhancement"""

    def __init__(self, in_channels=1, out_channels=1, base_filters=64, depth=4, dropout=0.2):
        super(UNet, self).__init__()

        self.depth = depth
        self.base_filters = base_filters

        # Encoder (Contracting path)
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels

        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoder_blocks.append(DownBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch * 2, dropout=dropout)

        # Decoder (Expanding path)
        self.decoder_blocks = nn.ModuleList()
        in_ch = in_ch * 2

        for i in range(depth):
            out_ch = base_filters * (2 ** (depth - i - 1))
            self.decoder_blocks.append(UpBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch

        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        # Store skip connections
        skips = []

        # Encoder path
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]  # Reverse order
            x = decoder_block(x, skip)

        # Final output
        x = self.final_conv(x)

        return x


class SpectralLoss(nn.Module):
    """Combined loss for spectrogram-based speech enhancement"""

    def __init__(self, mse_weight=1.0, l1_weight=0.1, spectral_weight=0.1):
        super(SpectralLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def spectral_convergence_loss(self, pred, target):
        """Spectral convergence loss"""
        return torch.norm(pred - target, p='fro') / torch.norm(target, p='fro')

    def forward(self, pred, target):
        # MSE loss (primary)
        mse = self.mse_loss(pred, target)

        # L1 loss (for sparsity)
        l1 = self.l1_loss(pred, target)

        # Spectral convergence
        spectral = self.spectral_convergence_loss(pred, target)

        total_loss = (self.mse_weight * mse + 
                     self.l1_weight * l1 + 
                     self.spectral_weight * spectral)

        return total_loss, {
            'mse': mse.item(),
            'l1': l1.item(),
            'spectral': spectral.item(),
            'total': total_loss.item()
        }
