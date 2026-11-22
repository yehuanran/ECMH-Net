from thop import profile
import torch
import timm
# from replace_conv_with_fdconv import replace_conv_with_fdconv
from Mamba_ConvNext_SemanticGuidedDecoder import HybridMambaUNet
# 创建网络实例（与train.py中的模型创建过程一致）
# 1. 创建基础模型
model = timm.create_model('convnextv2_base', pretrained=False)

# 2. 替换conv层为FDConv层
# replace_conv_with_fdconv(model, kernel_num=32)

# 3. 加载预训练权重
model.load_state_dict(torch.load("convnextv2_base.bin", map_location='cpu'))

# 4. 包装成完整的分割模型
model = HybridMambaUNet(backbone=model, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise RuntimeError("Mamba models require a GPU to run. No CUDA device found.")
model.to(device)
# 定义输入张量（根据您的配置）
batch_size = 1
H, W = 512, 512  # 根据实际输入尺寸调整，常用512x512或1024x1024
input_tensor = torch.randn(batch_size, 3, H, W).to(device)  # RGB图像输入

# 计算 FLOPs
flops, params = profile(model, inputs=(input_tensor,), verbose=False)
gflops = flops / 1e9  # 转换为 GFLOPs
params_m = params / 1e6  # 转换为 M（百万）

print(f"Input Size: {H}x{W}")
print(f"GFLOPs: {gflops:.2f}")
print(f"Parameters: {params_m:.2f}M")
