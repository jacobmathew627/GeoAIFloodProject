import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """Attention mechanism to focus on important spatial regions"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ChannelAttention(nn.Module):
    """Channel attention to weight important features (LULC vs DEM)"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class AttentionUNet(nn.Module):
    """Enhanced U-Net with Attention mechanisms"""
    def __init__(self, n_channels=5, n_classes=1):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.inc = double_conv(n_channels, 64)
        self.ca1 = ChannelAttention(64)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.ca2 = ChannelAttention(128)
        
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.ca3 = ChannelAttention(256)
        
        # Bottleneck
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        
        # Decoder with Attention
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv1 = double_conv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv2 = double_conv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv3 = double_conv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder with channel attention
        x1 = self.inc(x)
        x1 = x1 * self.ca1(x1)
        
        x2 = self.down1(x1)
        x2 = x2 * self.ca2(x2)
        
        x3 = self.down2(x2)
        x3 = x3 * self.ca3(x3)
        
        # Bottleneck
        x4 = self.down3(x3)
        
        # Decoder with spatial attention
        d3 = self.up1(x4)
        x3 = self.att1(g=d3, x=x3)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv1(d3)
        
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=x2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.up3(d2)
        x1 = self.att3(g=d1, x=x1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv3(d1)
        
        logits = self.outc(d1)
        return self.sigmoid(logits)

# Keep original UNet for comparison
class UNet(nn.Module):
    def __init__(self, n_channels=5, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.inc = double_conv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = double_conv(64, 32)
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)
        
        logits = self.outc(x)
        return self.sigmoid(logits)

if __name__ == "__main__":
    # Test both models
    print("Testing Attention U-Net...")
    att_net = AttentionUNet(n_channels=5, n_classes=1)
    x = torch.randn(2, 5, 32, 32)
    y = att_net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in att_net.parameters()):,}")
    
    print("\nTesting Original U-Net...")
    net = UNet(n_channels=5, n_classes=1)
    y2 = net(x)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
