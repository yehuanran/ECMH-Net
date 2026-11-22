import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

"""
混合架构：CNN + Mamba 可配置变体

通过 variant 参数控制每个 Stage 使用 CNN 还是 Mamba：
  '0' = CNN (使用 backbone 原始 stage)
  '1' = Mamba (使用 HybridMambaBlock 替换)

Variant | Stage 1 H/4 | Stage 2 H/8 | Stage 3 H/16 | Stage 4 H/32 | 描述
------- | ----------- | ----------- | ------------ | ------------ | --------------
A0      | CNN         | CNN         | CNN          | CNN          | 纯 CNN baseline
A1      | Mamba       | CNN         | CNN          | CNN          | 浅层 Mamba
A2      | CNN         | Mamba       | CNN          | CNN          | 中浅层 Mamba
A3      | CNN         | CNN         | Mamba        | CNN          | 深层 Mamba
A4      | CNN         | CNN         | CNN          | Mamba        | 最深层 Mamba
A5      | CNN         | CNN         | Mamba        | Mamba        | 你的方案
A6      | Mamba       | Mamba       | CNN          | CNN          | 浅层双 Mamba
A7      | Mamba       | Mamba       | Mamba        | Mamba        | All-Mamba

## ConvNeXtV2 各阶段通道数配置：
base:   [128, 256, 512, 1024]
large:  [192, 384, 768, 1536]
tiny:   [96, 192, 384, 768]
"""

# 各方案对应的 variant 编码
VARIANT_MAP = {
    'A0': '0000',
    'A1': '1000',
    'A2': '0100',
    'A3': '0010',
    'A4': '0001',
    'A5': '0011',
    'A6': '1100',
    'A7': '1111',
}


class HybridMambaBlock(nn.Module):
    """混合 Mamba 模块：结合 CNN 局部特征和 Mamba 全局上下文"""

    def __init__(self, dim):
        super().__init__()

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

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

        x_local = self.local_conv(x) + x

        x_flat = x_local.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)

        x_mamba = self.mamba(x_flat)

        x_mamba_2d = x_mamba.transpose(1, 2).reshape(B, C, H, W)

        x_out = x_mamba_2d + x_local

        return x_out


class MambaStage(nn.Module):
    """
    用 HybridMambaBlock 替换 backbone 的某个 stage。
    自动处理下采样：
      - stage_idx=0: 不下采样（与 backbone stages[0] 一致）
      - stage_idx>0: 先 2x 下采样（与 backbone stages[1,2,3] 一致）
    """

    def __init__(self, in_channels, out_channels, num_blocks, stage_idx):
        super().__init__()
        layers = []
        if stage_idx > 0:
            layers.append(nn.GroupNorm(1, in_channels))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2))
        else:
            assert in_channels == out_channels, \
                f"Stage 0 不下采样，in_channels({in_channels}) 应等于 out_channels({out_channels})"
        layers.extend([HybridMambaBlock(out_channels) for _ in range(num_blocks)])
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class SemanticGuidedDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // 2, skip_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            alpha = self.attention_gate(x)
            skip_filtered = skip * alpha

            x = torch.cat([x, skip_filtered], dim=1)

        x = self.fusion_conv(x)
        return x

class HybridMambaUNet(nn.Module):
    """
    混合架构：CNN + Mamba 可配置变体

    Args:
        backbone: timm 创建的 ConvNeXtV2 预训练模型
        num_classes: 分割类别数
        variant: 变体编号，可选 'A0'-'A7'，或直接用 4 位二进制字符串如 '0011'
    """

    def __init__(self, backbone, num_classes=2, variant='A5'):
        super(HybridMambaUNet, self).__init__()

        # 解析 variant
        if variant in VARIANT_MAP:
            self.variant_code = VARIANT_MAP[variant]
        else:
            assert len(variant) == 4 and all(c in '01' for c in variant), \
                f"variant 必须是 'A0'-'A7' 或 4 位 '01' 字符串，got: {variant}"
            self.variant_code = variant

        self.backbone = backbone
        self.stem = backbone.stem
        self.stages = backbone.stages

        # 通过前向传播获取各层通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            stem_out = self.stem(dummy_input)
            stage_outs = []
            prev = stem_out
            for i in range(4):
                prev = self.stages[i](prev)
                stage_outs.append(prev)

            self.stem_channels = stem_out.size(1)
            # stage 输入/输出通道数
            # stage_in_ch[i] = 进入 stage i 的通道数
            self.stage_in_ch = [stem_out.size(1)] + [stage_outs[i].size(1) for i in range(3)]
            # stage_out_ch[i] = stage i 输出的通道数
            self.stage_out_ch = [stage_outs[i].size(1) for i in range(4)]

            # 获取每个 stage 的 block 数量
            self.stage_num_blocks = [
                sum(1 for _ in self.stages[i].modules() if type(_).__name__ in ('ConvNeXtV2Block', 'ConvNeXtBlock'))
                for i in range(4)
            ]

        # feature_channels 保持与原来一致的索引方式（供 decoder 使用）
        self.feature_channels = [
            int(stage_outs[0].size(1) / 4),   # [0]: stem/4
            int(stage_outs[0].size(1) / 2),   # [1]: stem/2
            stage_outs[0].size(1),             # [2]: stage1 输出
            stage_outs[1].size(1),             # [3]: stage2 输出
            stage_outs[2].size(1),             # [4]: stage3 输出
            stage_outs[3].size(1)              # [5]: stage4 输出
        ]

        # ==================== 编码器 (Encoder) ====================
        # 根据 variant_code 决定每个 stage 用 CNN 还是 Mamba
        self.encoder_stages = nn.ModuleList()
        for i in range(4):
            if self.variant_code[i] == '1':
                # 用 Mamba 替换
                self.encoder_stages.append(
                    MambaStage(
                        in_channels=self.stage_in_ch[i],
                        out_channels=self.stage_out_ch[i],
                        num_blocks=self.stage_num_blocks[i],
                        stage_idx=i
                    )
                )
            else:
                # 使用 backbone 原始 CNN stage
                self.encoder_stages.append(self.stages[i])

        # ==================== 解码器 (Decoder) ====================
        self.dec4 = SemanticGuidedDecoderBlock(in_channels=self.feature_channels[5],
                                               skip_channels=self.feature_channels[4],
                                               out_channels=self.feature_channels[4])
        self.dec3 = SemanticGuidedDecoderBlock(in_channels=self.feature_channels[4],
                                               skip_channels=self.feature_channels[3],
                                               out_channels=self.feature_channels[3])
        self.dec2 = SemanticGuidedDecoderBlock(in_channels=self.feature_channels[3],
                                               skip_channels=self.feature_channels[2],
                                               out_channels=self.feature_channels[2])
        fuse_in_channels = self.feature_channels[2] + self.stem_channels
        self.fuse_x0 = nn.Sequential(
            nn.Conv2d(fuse_in_channels, self.feature_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels[1], self.feature_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_channels[1], self.feature_channels[1] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.feature_channels[1] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_channels[1] // 2, self.feature_channels[1] // 4, kernel_size=2, stride=2),
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
            nn.Conv2d(self.feature_channels[1] // 8, 1, kernel_size=1)  # 输出 1 通道的边缘概率图
        )
        # self.final_conv = nn.Conv2d(self.feature_channels[1] // 4, num_classes, kernel_size=1)

    def forward(self, x):
        # ==================== 编码阶段 ====================
        x0 = self.stem(x)

        # 4 个 stage，根据 variant 自动选择 CNN 或 Mamba
        feats = []
        prev = x0
        for i in range(4):
            prev = self.encoder_stages[i](prev)
            feats.append(prev)

        x1, x2, x3, x4 = feats

        # ==================== 解码阶段 ====================
        d4 = self.dec4(x4, skip=x3)
        d3 = self.dec3(d4, skip=x2)
        d2 = self.dec2(d3, skip=x1)

        d2_fused = torch.cat([d2, x0], dim=1)
        d2_fused = self.fuse_x0(d2_fused)

        d1 = self.dec1(d2_fused)
        # 双分支输出：面(Mask) + 线(Edge)
        out_mask = self.mask_head(d1)
        out_edge = self.edge_head(d1)

        if self.training:
            return out_mask, out_edge
        else:
            return out_mask


# # 测试代码
# if __name__ == '__main__':
#     import timm
#
#     dummy_input = torch.randn(2, 3, 512, 512)
#
#     for name, code in VARIANT_MAP.items():
#         backbone = timm.create_model('convnextv2_base', pretrained=False)
#         model = HybridMambaUNet(backbone, num_classes=2, variant=name)
#
#         output = model(dummy_input)
#         params = sum(p.numel() for p in model.parameters()) / 1e6
#
#         stage_types = ['Mamba' if c == '1' else 'CNN' for c in code]
#         print(f"{name} ({code}) | {stage_types[0]:5s} | {stage_types[1]:5s} | {stage_types[2]:5s} | {stage_types[3]:5s} | "
#               f"Params: {params:.2f}M | Output: {output.shape}")
