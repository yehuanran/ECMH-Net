import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
#这个是包括双头分支的解码器，完整版
"""
混合架构：浅层CNN + 深层Mamba
- Stage 1 & 2 (浅层): 使用 ConvNeXt V2 原始 Block 提取局部特征（边缘、纹理）
- Stage 3 & 4 (深层): 使用 HybridMambaBlock 提取全局语义信息
- Decoder: U-Net 风格的跳跃连接 + 反卷积恢复分辨率

## ConvNeXtV2 各阶段通道数配置：
base:   [128, 256, 512, 1024]
large:  [192, 384, 768, 1536]
tiny:   [96, 192, 384, 768]
"""


class SemanticGuidedDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 1. 上采样模块 (依然用转置卷积)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 2. 创新点 A：通道注意力引导 (Channel Attention Guide)
        # 用深层特征 (in_channels // 2) 生成一个权重，去筛选浅层特征 (skip_channels) 中的有用信息
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 2, skip_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. 特征融合与边界增强模块
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 引入深度可分离卷积，增强对建筑物边缘的局部感知，且不增加太多参数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # 深层特征上采样
        x = self.up(x)

        if skip is not None:
            # 确保尺寸一致
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # 创新点 A 的应用：用深层语义指导浅层细节
            # 生成注意力权重矩阵
            alpha = self.attention_gate(x)
            # 过滤跳跃连接中的冗余背景信息 (比如树木阴影)
            skip_filtered = skip * alpha

            # 拼接
            x = torch.cat([x, skip_filtered], dim=1)

        # 融合与特征重构
        x = self.fusion_conv(x)
        return x
#没有提升反而下降，不用了
class CrossPedigreeInteractionDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 1. 深层特征上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 2. 创新点：双流交互门控 (Dual-Stream Interaction Gate)
        # 空间注意力流：用于定位建筑物大致区域，过滤掉远处的道路、农田噪声
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 通道/高频引导流：用于捕获屋顶边缘突变的高频信号
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 2, skip_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # x: 深层特征 (来自Mamba分支)
        # skip: 浅层特征 (来自ConvNeXt分支)
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # A. 空间门控：强迫浅层特征聚焦于深层模型认为“有建筑”的区域
            s_attn = self.spatial_gate(x)
            skip_spatial = skip * s_attn

            # B. 通道门控：动态校准浅层特征的频道响应
            c_attn = self.channel_gate(x)
            skip_channel = skip * c_attn

            # 复合过滤
            skip_filtered = skip_spatial + skip_channel

            # 拼接
            x = torch.cat([x, skip_filtered], dim=1)

        return self.fusion(x)
class HybridMambaBlock(nn.Module):
    """混合 Mamba 模块：结合 CNN 局部特征和 Mamba 全局上下文"""

    def __init__(self, dim):
        super().__init__()

        # 分支1: 局部特征提取 - 深度可分离卷积抓取建筑物边缘
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 分支2: 全局特征提取 - Mamba 模块
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 提取局部边缘特征 (CNN)
        x_local = self.local_conv(x) + x

        # 2. 准备送入 Mamba (2D 转 1D)
        x_flat = x_local.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_flat = self.norm1(x_flat)

        # 3. 提取全局上下文 (Mamba)
        x_mamba = self.mamba(x_flat)

        # 4. 将 Mamba 输出还原回 2D 图像特征图
        x_mamba_2d = x_mamba.transpose(1, 2).reshape(B, C, H, W)

        # 5. 特征融合：全局视野 + 局部边缘
        x_out = x_mamba_2d + x_local

        return x_out

# 解码器基础块：上采样 + 拼接 + 卷积融合
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 使用转置卷积进行上采样 (将特征图放大 2 倍)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 融合跳跃连接过来的特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # 确保尺寸一致 (处理奇数分辨率的情况)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)  # 在通道维度拼接
        x = self.conv(x)
        return x
class StemDetailReinjectionBlock(nn.Module):
    """
    茎部细节再注入块：在 1/4 尺度将解码器语义特征 d2 与 stem 高分辨率细节特征 x0
    拼接并融合，补偿经多次下采样/上采样丢失的边缘与纹理细节。
    """
    def __init__(self, dec_channels, stem_channels, out_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(dec_channels + stem_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_dec, x_stem):
        x = torch.cat([x_dec, x_stem], dim=1)
        return self.fusion(x)

class HybridMambaUNet(nn.Module):
    """
    混合架构：浅层CNN + 深层Mamba

    Args:
        backbone: timm 创建的 ConvNeXtV2 预训练模型
        num_classes: 分割类别数
    """

    def __init__(self, backbone, num_classes=2):
        super(HybridMambaUNet, self).__init__()

        # 加载预训练的模型
        self.backbone = backbone

        # 获取特征提取层
        self.stem = backbone.stem
        self.stages = backbone.stages
        self.feature_channels = []
        # 通过实际前向传播确定各层的输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)

            # 获取stem输出
            stem_out = self.stem(dummy_input)

            # 获取各stage输出
            stage1_out = self.stages[0](stem_out)
            stage2_out = self.stages[1](stage1_out)
            stage3_out = self.stages[2](stage2_out)
            stage4_out = self.stages[3](stage3_out)
            # 获取各层的通道数,以base为例
            self.stem_channels = stem_out.size(1)  # 记录 stem 输出通道数 (通常为 128)
            # 获取各层的通道数,以base为例
            self.feature_channels = [
                                      int(stage1_out.size(1) / 4),  # 32
                                      int(stage1_out.size(1) / 2),  # 64
                                      stage1_out.size(1),  # stage1 输出通道数 128
                                      stage2_out.size(1),  # stage2 输出通道数 256
                                      stage3_out.size(1),  # stage3 输出通道数 512
                                      stage4_out.size(1)  # stage4 输出通道数 1024
                                    ]

        # ==================== 编码器 (Encoder) ====================

        # Stage 3: 深层 - 使用 HybridMambaBlock (全局语义理解)
        self.stage3 = nn.Sequential(
            nn.GroupNorm(1, self.feature_channels[3]), # 替换 LayerNorm 以支持 4D 输入 (B, C, H, W)
            nn.Conv2d(self.feature_channels[3], self.feature_channels[4], kernel_size=2, stride=2),
            *[HybridMambaBlock(self.feature_channels[4]) for _ in range(9)]
        )

        # Stage 4: 深层 - 使用 HybridMambaBlock
        self.stage4 = nn.Sequential(
            nn.GroupNorm(1, self.feature_channels[4]), # 替换 LayerNorm 以支持 4D 输入 (B, C, H, W)
            nn.Conv2d(self.feature_channels[4], self.feature_channels[5], kernel_size=2, stride=2),
            *[HybridMambaBlock(self.feature_channels[5]) for _ in range(3)]
        )

        # ==================== 解码器 (Decoder) ====================

        # --- 解码器 (Decoder) ---
        # 尺寸恢复：1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/1

        self.dec4 = SemanticGuidedDecoderBlock(in_channels=self.feature_channels[5], skip_channels=self.feature_channels[4], out_channels=self.feature_channels[4])  # 1/32 to 1/16
        self.dec3 = SemanticGuidedDecoderBlock(in_channels=self.feature_channels[4], skip_channels=self.feature_channels[3], out_channels=self.feature_channels[3])  # 1/16 to 1/8
        self.dec2 = SemanticGuidedDecoderBlock(in_channels=self.feature_channels[3], skip_channels=self.feature_channels[2], out_channels=self.feature_channels[2])  # 1/8 to 1/4
        # 新增：在 1/4 尺度融合 x0 浅层特征（先调整通道再拼接）
        # d2 通道数为 feature_channels[2], x0 通道数为 stem_channels
        fuse_in_channels = self.feature_channels[2] + self.stem_channels
        self.fuse_x0 = StemDetailReinjectionBlock(
            dec_channels=self.feature_channels[2],
            stem_channels=self.stem_channels,
            out_channels=self.feature_channels[1]
        )
        # fuse_in_channels = self.feature_channels[2] + self.stem_channels
        # self.fuse_x0 = nn.Sequential(
        #     nn.Conv2d(fuse_in_channels, self.feature_channels[1], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.feature_channels[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.feature_channels[1], self.feature_channels[1], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.feature_channels[1]),
        #     nn.ReLU(inplace=True)
        # )

        # 从 1/4 恢复到原图大小
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_channels[1], self.feature_channels[1] // 2, kernel_size=2, stride=2),
            # 1/4 -> 1/2
            nn.BatchNorm2d(self.feature_channels[1] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_channels[1] // 2, self.feature_channels[1] // 4, kernel_size=2, stride=2),
            # 1/2 -> 1
            nn.BatchNorm2d(self.feature_channels[1] // 4),
            nn.ReLU(inplace=True),
        )
        # 创新点 B：双分支输出头 (Dual-Head)
        # 1. 语义掩码预测头 (输出建筑物的面)
        self.mask_head = nn.Conv2d(self.feature_channels[1] // 4, num_classes, kernel_size=1)

        # 2. 边界约束头 (输出建筑物的轮廓线)
        self.edge_head = nn.Sequential(
            nn.Conv2d(self.feature_channels[1] // 4, self.feature_channels[1] // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels[1]// 8, 1, kernel_size=1)  # 输出 1 通道的边缘概率图
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            分割结果 (B, num_classes, H, W)
        """

        # ==================== 编码阶段 ====================
        # Stage 0: 浅层特征 (高分辨率，提取边缘/纹理)
        x0 = self.stem(x)

        # Stage 1: 浅层特征 (高分辨率，提取边缘/纹理)
        x1 = self.stages[0](x0)

        # Stage 2: 浅层特征
        x2 = self.stages[1](x1)  # (B, C2, H/8, W/8)

        # Stage 3: 深层特征 (低分辨率，提取全局语义)
        x3 = self.stage3(x2)  # (B, C3, H/16, W/16)

        # Stage 4: 深层特征
        x4 = self.stage4(x3)  # (B, C4, H/32, W/32)

        # ---------------- 解码过程 ----------------
        # f4 结合 f3 (Mamba 层的互相纠错)
        d4 = self.dec4(x4, skip=x3)

        # 结合 f2 (融入 CNN 提取的高频特征)
        d3 = self.dec3(d4, skip=x2)

        # 结合 f1 (极其精细的边缘信息)
        d2 = self.dec2(d3, skip=x1)

        # 在 1/4 尺度融合 x0 浅层特征（d2 和 x0 都是 H/4, W/4，可以直接拼接）
        # d2_fused = torch.cat([d2, x0], dim=1)  # 在通道维度拼接
        # d2_fused = self.fuse_x0(d2_fused)
        d2_fused = self.fuse_x0(d2, x0)
        # 最终恢复到原图大小 - 1/4 -> 1/2 -> 1
        d1 = self.dec1(d2_fused)

        # 分别输出 面(Mask) 和 线(Edge)
        out_mask = self.mask_head(d1)
        out_edge = self.edge_head(d1)

        # 如果是训练阶段，同时返回 mask 和 edge 用于计算联合 Loss；
        # 如果是推理阶段，只返回 mask 即可。
        if self.training:
            return out_mask, out_edge
        else:
            return out_mask


# 测试代码
if __name__ == '__main__':
    import timm

    # 创建模型
    backbone = timm.create_model('convnextv2_base', pretrained=False)
    model = HybridMambaUNet(backbone, num_classes=2)

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)

    print(f"输入维度: {dummy_input.shape}")
    print(f"输出维度: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
