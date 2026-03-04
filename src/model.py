import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels=9, n_classes=1):
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
        # x: (B, 6, H, W)
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
    net = UNet(n_channels=6)
    print(net)
    x = torch.randn(1, 6, 32, 32)
    y = net(x)
    print("Output shape:", y.shape)
