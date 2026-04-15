import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 稀疏注意力模块 =====
class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=4, sparsity=0.5, mode='topk', local_window=5):
        super(SparseAttention, self).__init__()
        assert dim % num_heads == 0
        assert mode in ['random', 'topk', 'local']

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity = sparsity
        self.mode = mode
        self.local_window = local_window

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: [B, C, H, W]
        """
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.flatten(2).transpose(1, 2)  # [B, L, C]

        q = self.q_proj(x_flat).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # 稀疏掩码
        if self.mode == 'random':
            mask = torch.bernoulli(self.sparsity * torch.ones_like(attn_scores))
        elif self.mode == 'topk':
            k_val = max(1, int(L * self.sparsity))
            topk_mask = torch.zeros_like(attn_scores)
            _, topk_indices = torch.topk(attn_scores, k=k_val, dim=-1)
            topk_mask.scatter_(-1, topk_indices, 1.0)
            mask = topk_mask
        elif self.mode == 'local':
            mask = torch.zeros_like(attn_scores)
            for i in range(L):
                left = max(0, i - self.local_window)
                right = min(L, i + self.local_window + 1)
                mask[:, :, i, left:right] = 1.0

        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


# ===== 改进后的 FusionModule =====
class FusionModule(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, use_sparse_attn=True):
        super(FusionModule, self).__init__()
        self.use_sparse_attn = use_sparse_attn

        # 池化下采样
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 深度可分离卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96),
            nn.Conv2d(96, 64, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192),
            nn.Conv2d(192, 128, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=384),
            nn.Conv2d(384, 256, kernel_size=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=768),
            nn.Conv2d(768, 512, kernel_size=1)
        )

        self.relu = nn.ReLU()

        # 通道融合卷积
        from changedetection.models.ChangeDecoder import Conv
        self.cv1 = Conv(960, 512)
        self.cv2 = Conv(512, 256)

        # 稀疏注意力模块（可选）
        if use_sparse_attn:
            self.sparse_attn = SparseAttention(dim=512, num_heads=4, sparsity=0.3, mode='local')

        # 多尺度输出
        self.out1 = Conv(512, 128)
        self.out2 = Conv(512, 128)
        self.out3 = Conv(512, 128)
        self.out4 = Conv(512, 128)

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # pre 路径
        pr1_down = self.relu(self.conv1(self.pool1(pre_feat_1)))
        pr2_down = self.relu(self.conv2(self.pool2(pre_feat_2)))
        pr3_up = self.relu(self.conv3(pre_feat_3))
        pr4_up = self.relu(self.conv4(F.interpolate(pre_feat_4, scale_factor=2, mode='bilinear', align_corners=True)))

        # post 路径
        po1_down = self.relu(self.conv1(self.pool1(post_feat_1)))
        po2_down = self.relu(self.conv2(self.pool2(post_feat_2)))
        po3_up = self.relu(self.conv3(post_feat_3))
        po4_up = self.relu(self.conv4(F.interpolate(post_feat_4, scale_factor=2, mode='bilinear', align_corners=True)))

        # 特征拼接与差异计算
        pre_fused = torch.cat([pr1_down, pr2_down, pr3_up, pr4_up], dim=1)
        post_fused = torch.cat([po1_down, po2_down, po3_up, po4_up], dim=1)
        x1 = torch.abs(pre_fused - post_fused)
        x1 = self.cv1(x1)

        # # 稀疏注意力增强特征
        # if self.use_sparse_attn:
        #     x1 = self.sparse_attn(x1)

        # 多尺度输出
        pr1 = self.out1(F.interpolate(x1, scale_factor=4, mode='bilinear'))
        pr2 = self.out2(F.interpolate(x1, scale_factor=2, mode='bilinear'))
        pr3 = self.out3(x1)
        pr4 = self.out4(F.max_pool2d(x1, 2))

        return pr1, pr2, pr3, pr4

class FusionModuleUp(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, use_sparse_attn=True):
        super(FusionModuleUp, self).__init__()
        self.use_sparse_attn = use_sparse_attn

        # 池化下采样
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 深度可分离卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96),
            nn.Conv2d(96, 64, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192),
            nn.Conv2d(192, 128, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=384),
            nn.Conv2d(384, 256, kernel_size=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=768),
            nn.Conv2d(768, 512, kernel_size=1)
        )

        self.relu = nn.ReLU()

        # 通道融合卷积
        from changedetection.models.ChangeDecoder import Conv
        self.cv1 = Conv(960, 512)
        self.cv2 = Conv(512, 256)

        # 稀疏注意力模块（可选）
        if use_sparse_attn:
            self.sparse_attn = SparseAttention(dim=512, num_heads=4, sparsity=0.3, mode='local')

        # 多尺度输出
        self.out1 = Conv(512, 128)
        self.out2 = Conv(512, 128)
        self.out3 = Conv(512, 128)
        self.out4 = Conv(512, 128)

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # pre 路径
        pr1_down = self.relu(self.conv1(pre_feat_1))
        pr2_down = self.relu(self.conv2(F.interpolate(pre_feat_2, scale_factor=2, mode='bilinear', align_corners=True)))
        pr3_up = self.relu(self.conv3(F.interpolate(pre_feat_3, scale_factor=4, mode='bilinear', align_corners=True)))
        pr4_up = self.relu(self.conv4(F.interpolate(pre_feat_4, scale_factor=8, mode='bilinear', align_corners=True)))

        # post 路径
        po1_down = self.relu(self.conv1(post_feat_1))
        po2_down = self.relu(self.conv2(F.interpolate(post_feat_2, scale_factor=2, mode='bilinear', align_corners=True)))
        po3_up = self.relu(self.conv3(F.interpolate(post_feat_3, scale_factor=4, mode='bilinear', align_corners=True)))
        po4_up = self.relu(self.conv4(F.interpolate(post_feat_4, scale_factor=8, mode='bilinear', align_corners=True)))

        # 特征拼接与差异计算
        pre_fused = torch.cat([pr1_down, pr2_down, pr3_up, pr4_up], dim=1)
        post_fused = torch.cat([po1_down, po2_down, po3_up, po4_up], dim=1)
        x1 = torch.abs(pre_fused - post_fused)
        x1 = self.cv1(x1)

        # # 稀疏注意力增强特征
        # if self.use_sparse_attn:
        #     x1 = self.sparse_attn(x1)

        # 多尺度输出
        pr1 = self.out1(x1)
        pr2 = self.out2(F.max_pool2d(x1, 2))
        pr3 = self.out3(F.max_pool2d(x1, 4))
        pr4 = self.out4(F.max_pool2d(x1, 8))
        #print(pr1.shape, pr2.shape, pr3.shape, pr4.shape)
        return pr1, pr2, pr3, pr4
