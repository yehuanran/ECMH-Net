import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """U-Net 基础块：(Conv3x3 + BN + ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net (Ronneberger et al., 2015)，用于对比实验 baseline
    输入: (B, 3, 512, 512)
    输出: (B, num_classes, 512, 512) 的 Logits（不经过 Softmax）
    """
    def __init__(self, in_channels=3, num_classes=2, base_channels=64):
        super().__init__()
        c = base_channels  # 默认 64，显存不够可改为 32

        # 编码器
        self.enc1 = DoubleConv(in_channels, c)        # (B, 64, 512, 512)
        self.enc2 = DoubleConv(c, c * 2)              # (B, 128, 256, 256)
        self.enc3 = DoubleConv(c * 2, c * 4)          # (B, 256, 128, 128)
        self.enc4 = DoubleConv(c * 4, c * 8)          # (B, 512, 64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = DoubleConv(c * 8, c * 16)   # (B, 1024, 32, 32)

        # 解码器：转置卷积上采样 + 跳跃连接拼接 + 双卷积融合
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)         # (B, 512, 64, 64)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)          # (B, 256, 128, 128)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)          # (B, 128, 256, 256)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)              # (B, 64, 512, 512)

        # 分割头：1x1 卷积输出类别数
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def _cat(self, x, skip):
        # 确保尺寸一致（处理非 512 输入时的奇数分辨率）
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈
        b = self.bottleneck(self.pool(e4))

        # 解码器
        d4 = self.dec4(self._cat(self.up4(b), e4))
        d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))

        return self.head(d1)


if __name__ == '__main__':
    model = UNet(in_channels=3, num_classes=2)
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(out.shape)  # torch.Size([1, 2, 512, 512])
    print('参数量: {:.2f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))