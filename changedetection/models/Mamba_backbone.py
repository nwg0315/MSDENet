from timm.layers import DropPath

from classification.models.vmamba import VSSM, LayerNorm2d

import torch
import torch.nn as nn


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        # print("outs", outs[0].shape, outs[1].shape, outs[2].shape, outs[3].shape)
        return outs


class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, e=2):
        super().__init__()

        self.pw1 = nn.Conv2d(in_dim, in_dim * e, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_dim * e)
        self.act1 = nn.GELU()

        self.dw = nn.Conv2d(in_dim * e, in_dim * e, kernel_size=kernel, stride=1, padding=1, groups=in_dim * e)
        self.norm2 = nn.BatchNorm2d(in_dim * e)
        self.act2 = nn.GELU()

        self.pw2 = nn.Conv2d(in_dim * e, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)
        self.dim = in_dim

    def forward(self, x):
        # print(self.dim)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        # print(x.shape)
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.pw2(x)
        x = self.norm3(x)
        x = x.permute(0, 3, 2, 1)  # (B, C, H, W)
        # print(x.shape)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel=3, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.dim = dim
        # print('dim', dim)
        self.kernel = kernel
        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        # print('inc1',self.dim)
        # print(self.kernel)
        # print(self.use_layer_scale)
        # print('invert',self.layer_scale_1.shape)
        # if self.use_layer_scale:
        #     x = x + self.drop_path(self.layer_scale_1 * self.dws(x))
        # else:
        x1 = self.dws(x)
        x = x + self.drop_path(x1)
        # print('okIR')
        # x = x.permute(0, 3, 2, 1)  # (B, C, H, W)
        # print('ir',x.shape)
        return x


class Backbone_VSSM1(nn.Module):
    def __init__(self, out_indices=(0, 1, 2, 3), dims=[32, 64, 128, 256],
                 depths=[2, 2, 9, 2], pretrained=None, norm_layer='ln2d',
                 patch_size=4, **kwargs):  # 添加 patch_size 和 **kwargs
        super().__init__()
        if isinstance(dims, int):
            dims = [dims * (2 ** i) for i in range(4)]  # 例如 96 -> [96, 192, 384, 768]
        self.dims = dims
        self.depths = depths
        self.out_indices = out_indices
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=patch_size, stride=patch_size),  # 使用 patch_size
            #nn.LayerNorm(dims[0]),
        )

        # 初始化 norm_layer
        _NORMLAYERS = {
            'ln': nn.LayerNorm,
            'ln2d': LayerNorm2d,
            'bn': nn.BatchNorm2d,
        }
        norm_layer = _NORMLAYERS.get(norm_layer.lower(), nn.LayerNorm)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([dims[0], 64, 64]),  # 明确指定归一化维度 (C, H, W)
        )

        # 构建层级结构
        self.layers = nn.ModuleList()
        for i in range(len(dims)):
            layer = nn.Module()
            # 用倒残差块替换 Mamba 块
            layer.blocks = nn.Sequential(*[
                InvertedResidual(dims[i]) for _ in range(depths[i])
            ])
            # 下采样（跨步卷积）
            if i < len(dims) - 1:
                layer.downsample = nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            else:
                layer.downsample = nn.Identity()
            self.layers.append(layer)

        # 输出归一化层
        for i in out_indices:
            layer = norm_layer(dims[i])
            self.add_module(f'outnorm{i}', layer)

        # 加载预训练权重
        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location="cpu")
            self.load_state_dict(_ckpt[key], strict=False)
            print(f"Loaded pretrained weights from {ckpt}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    def forward(self, x):
        # 输入 x: (B, 3, H, W)
        x = self.patch_embed(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        outs = []
        for i, layer in enumerate(self.layers):
            # 倒残差块处理
            x = layer.blocks(x)  # (B, H, W, C)
            # 下采样
            y = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            y = layer.downsample(y)  # (B, C_new, H_new, W_new)
            y = y.permute(0, 2, 3, 1)  # (B, H_new, W_new, C_new)

            # 保存当前层输出
            if i in self.out_indices:
                out = getattr(self, f'outnorm{i}')(x)
                # if self.channel_first:
                out = out.permute(0, 3, 1, 2)
                outs.append(out)
            x = y  # 更新 x 为下采样结果
        # print('isisi', outs[0].shape)
        # x = x.permute(0, 3, 2, 1)  # (B, C, H, W)
        return outs if len(self.out_indices) > 0 else x