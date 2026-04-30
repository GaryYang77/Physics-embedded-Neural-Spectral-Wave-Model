import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class DoubleConv(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU."""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


@gin.configurable
class Down(nn.Module):
    """Downsampling block with AvgPool2d + DoubleConv."""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


@gin.configurable  
class Up(nn.Module):
    """Upsampling block with transpose conv or bilinear resize + DoubleConv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@gin.configurable
class OutConv(nn.Module):
    """Output projection layer."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


@gin.configurable
class UNet(nn.Module):
    """
    UNet with configurable depth.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_channels: Base channel width.
        depth: Encoder depth.
        bilinear: Whether to use bilinear upsampling.
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, depth=4, bilinear=True):
        super(UNet, self).__init__()
        self.depth = depth
        self.bilinear = bilinear

        self.channels = [base_channels]
        for i in range(depth):
            self.channels.append(base_channels * (2 ** (i + 1)))

        self.inc = DoubleConv(in_channels, self.channels[0])

        self.down_layers = nn.ModuleList()
        for i in range(depth):
            self.down_layers.append(Down(self.channels[i], self.channels[i + 1]))

        self.up_layers = nn.ModuleList()
        for i in range(depth):
            up_in_channels = self.channels[depth - i] + self.channels[depth - i - 1]
            up_out_channels = self.channels[depth - i - 1]
            self.up_layers.append(Up(up_in_channels, up_out_channels, bilinear))

        self.outc = OutConv(self.channels[0], out_channels)
    
    def forward(self, x):
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)

        for down in self.down_layers:
            x = down(x)
            skip_connections.append(x)

        skip_connections.pop()

        for up in self.up_layers:
            skip_x = skip_connections.pop()
            x = up(x, skip_x)

        logits = self.outc(x)
        return logits
