import torch
from torch import nn


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        #print(x1.shape)
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        return out_x1, out_x2

import torch
from torch import nn
import torch.nn.functional as F


class LearnableChannelExchange(nn.Module):
    """
        通道交换模块（随机 + 比例可控版）
        ---------------------------------
        功能：
            在两个特征图间部分交换通道信息，以促进双分支特征融合。
        参数：
            ratio: float ∈ [0,1]，交换的通道比例（默认0.5）
            mode: str，'random'（训练时随机交换）或 'periodic'（固定间隔交换）
            p: int，当 mode='periodic' 时，表示每隔 p 个通道交换一次
        """

    def __init__(self, ratio=0.5, mode='random', p=2):
        super().__init__()
        self.ratio = ratio
        self.mode = mode
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        device = x1.device

        # --- 构造交换掩码 ---
        if self.mode == 'random' and self.training:
            # 随机生成交换掩码，按比例交换
            num_swap = int(C * self.ratio)
            idx = torch.randperm(C, device=device)[:num_swap]
            mask = torch.zeros(C, device=device, dtype=torch.bool)
            mask[idx] = True
        else:
            # 固定间隔交换（确定性模式）
            mask = (torch.arange(C, device=device) % self.p == 0)

        # reshape 成 [1, C, 1, 1] 以便广播
        mask = mask.view(1, C, 1, 1)

        # --- 执行通道交换 ---
        out_x1 = torch.where(mask, x2, x1)
        out_x2 = torch.where(mask, x1, x2)

        return out_x1, out_x2