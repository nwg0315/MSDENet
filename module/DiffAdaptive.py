import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# CBAM模块：包含通道注意力 + 空间注意力
# -----------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        # 空间注意力
        sa = self.spatial_attention(torch.cat([torch.mean(x, 1, keepdim=True),
                                               torch.max(x, 1, keepdim=True)[0]], dim=1))
        x = x * sa
        return x


# -----------------------------------------------------
# Adaptive Diff 模块：自适应差分（learnable difference weighting）
# -----------------------------------------------------
class AdaptiveDiffFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveDiffFusion, self).__init__()
        self.weight_gen = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, pre, post):
        diff = pre - post
        cat = pre+post
        w = self.weight_gen(cat)
        fused = w * diff + (1 - w) * (pre + post) / 2
        return fused


# -----------------------------------------------------
# 综合版融合模块：Adaptive + 双向差分 + CBAM
# -----------------------------------------------------
class BiAdaptiveCBAMFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super(BiAdaptiveCBAMFusion, self).__init__()
        self.adaptive_diff = AdaptiveDiffFusion(channels)

        # 输入通道：Adaptive diff + 双向 diff + pre + post
        in_ch = channels * 3  # [adaptive_diff, pre, post, pre-post, post-pre]
        self.cbam = CBAM(in_ch, reduction=reduction)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, pre, post):
        # Step1: Adaptive 差分
        adaptive_diff = self.adaptive_diff(pre, post)

        # Step2: 双向差分
        diff1 = pre - post
        diff2 = post - pre

        # Step3: 拼接并通过 CBAM 注意力
        fused = torch.cat([adaptive_diff, diff1, diff2], dim=1)
        fused = self.cbam(fused)

        # Step4: 降维输出
        out = self.out_conv(fused)
        return out


# -----------------------------------------------------
# ✅ 示例使用
# -----------------------------------------------------
if __name__ == "__main__":
    pre_feat = torch.randn(2, 64, 64, 64)
    post_feat = torch.randn(2, 64, 64, 64)

    fusion = BiAdaptiveCBAMFusion(channels=64)
    out = fusion(pre_feat, post_feat)
    print("输出特征尺寸:", out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# CBAM模块：包含通道注意力 + 空间注意力
# -----------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        # 空间注意力
        sa = self.spatial_attention(torch.cat([torch.mean(x, 1, keepdim=True),
                                               torch.max(x, 1, keepdim=True)[0]], dim=1))
        x = x * sa
        return x


# -----------------------------------------------------
# Adaptive Diff 模块：自适应差分（learnable difference weighting）
# -----------------------------------------------------
class AdaptiveDiffFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveDiffFusion, self).__init__()
        self.weight_gen = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, pre, post):
        diff = pre - post
        cat = pre+post
        w = self.weight_gen(cat)
        fused = w * diff + (1 - w) * (pre + post) / 2
        return fused


# -----------------------------------------------------
# 综合版融合模块：Adaptive + 双向差分 + CBAM
# -----------------------------------------------------
class BiAdaptiveCBAMFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super(BiAdaptiveCBAMFusion, self).__init__()
        self.adaptive_diff = AdaptiveDiffFusion(channels)

        # 输入通道：Adaptive diff + 双向 diff + pre + post
        in_ch = channels * 3  # [adaptive_diff, pre, post, pre-post, post-pre]
        self.cbam = CBAM(in_ch, reduction=reduction)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, pre, post):
        # Step1: Adaptive 差分
        adaptive_diff = self.adaptive_diff(pre, post)

        # Step2: 双向差分
        diff1 = pre - post
        diff2 = post - pre

        # Step3: 拼接并通过 CBAM 注意力
        fused = torch.cat([adaptive_diff, diff1, diff2], dim=1)
        fused = self.cbam(fused)

        # Step4: 降维输出
        out = self.out_conv(fused)
        return out


# -----------------------------------------------------
# ✅ 示例使用
# -----------------------------------------------------
if __name__ == "__main__":
    pre_feat = torch.randn(2, 64, 64, 64)
    post_feat = torch.randn(2, 64, 64, 64)

    fusion = BiAdaptiveCBAMFusion(channels=64)
    out = fusion(pre_feat, post_feat)
    print("输出特征尺寸:", out.shape)
