import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from changedetection.utils_func.lovasz_loss import lovasz_softmax


# Lovász Softmax 需要你已导入模块 L.lovasz_softmax
# 示例： from loss.lovasz_losses import lovasz_softmax


class AdaptiveBoundaryFocalLovaszLoss(nn.Module):
    """
    最优综合版损失函数：
    Focal Loss + Lovasz Softmax + Boundary Loss + 自适应加权
    """
    def __init__(self, alpha=0.25, gamma=2.0, boundary_weight=0.1, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

    def focal_loss(self, pred, target):
        """Focal Loss (for class imbalance)"""
        ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()

    def boundary_loss(self, pred, target):
        """Boundary-aware loss via Sobel gradient maps"""
        pred_probs = F.softmax(pred, dim=1)[:, 1:2]  # 假设二分类，取变化类通道
        target = target.float().unsqueeze(1)
        target_edges = kornia.filters.sobel(target)
        pred_edges = kornia.filters.sobel(pred_probs)
        return F.l1_loss(pred_edges, target_edges)

    def forward(self, pred, target):
        # --- 主体损失项 ---
        focal = self.focal_loss(pred, target)
        lovasz = lovasz_softmax(F.softmax(pred, dim=1), target, ignore=self.ignore_index)
        boundary = self.boundary_loss(pred, target)

        # --- 动态自适应加权 ---
        total_main = focal + lovasz + 1e-6
        wf = focal / total_main
        wl = lovasz / total_main

        loss = wf * focal + wl * lovasz + self.boundary_weight * boundary
        return loss


# Copyright 2023 University of Basel and Lucerne University of Applied Sciences and Arts Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


__author__ = "Alvaro Gonzalez-Jimenez"
__maintainer__ = "Alvaro Gonzalez-Jimenez"
__email__ = "alvaro.gonzalezjimenez@unibas.ch"
__license__ = "Apache License, Version 2.0"
__date__ = "2023-07-25"

import numpy as np
import torch
import torch.nn as nn


# NOTE: The mismatch between the paper and the code is because we offer a more general
# formulation where Sigma is an arbitrary diagonal matrix.
# For a detailed explanation, please refer to: https://github.com/Digital-Dermatology/t-loss/issues/2

class TLoss(nn.Module):
    def __init__(
            self,
            config,
            nu: float = 1.0,
            epsilon: float = 1e-8,
            reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        if config is not None:
            device = config.device
            image_size = config.data.image_size
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_size = 256  # 默认图像大小，可自行改

        self.config = config
        self.D = torch.tensor((image_size * image_size), dtype=torch.float, device=device)
        self.lambdas = torch.ones((image_size, image_size), dtype=torch.float, device=device)
        self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float, device=device))
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=device)
        self.reduction = reduction

    def forward(
            self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
            target_tensor (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """

        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
                first_term
                + second_term
                + third_term
                + fourth_term
                + fifth_term
                + sixth_term
        )

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )
class CombinedLoss(nn.Module):
    def __init__(self, config, t_weight=0.05):
        super(CombinedLoss, self).__init__()
        self.t_weight = t_weight
        self.t_loss_fn = TLoss(config=config, nu=1.0, reduction='mean')

    def forward(self, output_1, labels):
        # --- 原始损失部分 ---
        ce_loss_1 = F.cross_entropy(output_1, labels, ignore_index=255)
        lovasz_loss = lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
        main_loss = ce_loss_1 + 0.75 * lovasz_loss

        # --- TLoss部分 ---
        # 取预测概率的最大类（或者取softmax概率图再选取某通道），与标签对齐
        prob_map = F.softmax(output_1, dim=1)
        pred_mask = torch.argmax(prob_map, dim=1).float()
        gt_mask = labels.float()

        # 注意：TLoss期望输入 (B, W, H)
        t_loss_val = self.t_loss_fn(pred_mask, gt_mask)

        # --- 综合总损失 ---
        final_loss = main_loss + self.t_weight * t_loss_val
        return final_loss

class HybridLoss(nn.Module):
    def __init__(self, config, ce_weight=1.0, lovasz_weight=0.75, t_weight=0.5, ignore_index=255):
        super(HybridLoss, self).__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.t_weight = t_weight
        self.ignore_index = ignore_index

        # 初始化 T-Loss
        self.t_loss = TLoss(config, nu=1.0, epsilon=1e-8, reduction="mean")

    def forward(self, output_1, labels):
        """
        Args:
            output_1: [B, C, H, W] 网络预测结果
            labels:   [B, H, W] 标签
        """
        # ====== 1. CrossEntropy Loss ======
        ce_loss = F.cross_entropy(output_1, labels, ignore_index=self.ignore_index)

        # ====== 2. Lovasz Softmax Loss ======
        lovasz_loss = lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=self.ignore_index)

        # ====== 3. T-Loss (Robust Term) ======
        # 对于变化检测任务，可以选择“变化类”通道 (通常为1)
        prob_map = F.softmax(output_1, dim=1)[:, 1, :, :]  # 取变化类概率
        t_loss_val = self.t_loss(prob_map, labels.float())

        # ====== 4. 总损失融合 ======
        total_loss = (
            self.ce_weight * ce_loss +
            self.lovasz_weight * lovasz_loss +
            self.t_weight * t_loss_val
        )
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceAwareFocalLoss(nn.Module):
    def __init__(self, lambda_param=1.0, gamma=2.0, reduction='mean'):
        super(DistanceAwareFocalLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.reduction = reduction

    def distance_to_confidence(self, distance_loss):
        return torch.exp(-self.lambda_param * distance_loss)

    def forward(self, pred_logits, target_labels, distance_loss=None):
        # 仅支持二分类变化检测
        pred_probs = torch.sigmoid(pred_logits).squeeze()

        if distance_loss is not None:
            localization_conf = self.distance_to_confidence(distance_loss)
            continuous_labels = target_labels * localization_conf
        else:
            continuous_labels = target_labels

        pred_probs = torch.clamp(pred_probs, 1e-7, 1 - 1e-7)
        bce_term = -(continuous_labels * torch.log(pred_probs) +
                     (1 - continuous_labels) * torch.log(1 - pred_probs))
        modulation_factor = torch.abs(continuous_labels - pred_probs).pow(self.gamma)
        loss = modulation_factor * bce_term

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DistanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DistanceLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.reduction = reduction

    def forward(self, pred_points, gt_points):
        distance = self.l1_loss(pred_points, gt_points)
        if distance.dim() > 2:
            distance = distance.view(distance.size(0), -1).sum(dim=1)
        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        else:
            return distance
