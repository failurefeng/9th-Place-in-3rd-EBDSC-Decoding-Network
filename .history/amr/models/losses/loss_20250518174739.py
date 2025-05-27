import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class loss_CE(nn.Module):
    def __init__(self, thr_low=0.7, thr_high=0.95, penalty_weight=0.5, safety_margin=0.05):
        super().__init__()
        self.pad_value = 0.0
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
    def forward(self, targets, preds, mask):
        """
        Args:
            preds:      (B, S_max)  模型预测结果（包含填充），元素非整数
            targets:    (B, S_max)  真实标签（包含填充）
            mask:       (B, S_max)  实际有效位置掩码
            target_lengths: (B,) 每个样本的有效长度
        Returns:
            loss:       scalar 综合损失
            cq_scores:  (B,)  质量评分
        """
        mask = mask.to(preds.device) # (B,S_max)

        max_length = max(targets.size(1), preds.size(1), mask.size(1))
        preds_padded, targets_padded, mask_padded = [F.pad(tensor, (0, max_length - tensor.size(1)), value=0) for tensor in [preds, targets, mask]]

        useful_preds = preds_padded * mask_padded
        useful_targets = targets_padded * mask_padded
        
        # --- 余弦相似度计算 ---
        cos_sims = self.cos(useful_preds, useful_targets)
        
        rounded_cos_sims = self.cos(useful_preds.round(), useful_targets.round())

        total_loss = self.CE(preds_padded, targets_padded)
        
        # --- 质量评分计算 ---
        cq_scores = torch.where(
            cos_sims < 0.7, 
            0.0,
            torch.where(cos_sims <=0.95, 100*(cos_sims-0.7)/0.25, 100.0)
        )

        rounded_cq_scores = torch.where(
            rounded_cos_sims < 0.7,
            0.0,
            torch.where(rounded_cos_sims <=0.95, 100*(rounded_cos_sims-0.7)/0.25, 100.0)
        )

        if torch.isnan(loss_cos).any():
            raise ValueError("NaN detected in loss_cos")
        return total_loss, cq_scores.mean(), rounded_cq_scores.mean()

class loss_cos(nn.Module):
    def __init__(self, thr_low=0.7, thr_high=0.95, penalty_weight=0.5, safety_margin=0.05):
        super().__init__()
        self.pad_value = 0.0
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.alpha = 0.3
        self.thr_low = thr_low
        self.thr_high = thr_high
        self.penalty_weight = penalty_weight
        self.safety_margin = safety_margin  # 提前惩罚边界（防止卡在阈值临界点）
    def forward(self, targets, preds, mask):
        """
        Args:
            preds:      (B, S_max)  模型预测结果（包含填充），元素非整数
            targets:    (B, S_max)  真实标签（包含填充）
            mask:       (B, S_max)  实际有效位置掩码
            target_lengths: (B,) 每个样本的有效长度
        Returns:
            loss:       scalar 综合损失
            cq_scores:  (B,)  质量评分
        """
        mask = mask.to(preds.device) # (B,S_max)

        max_length = max(targets.size(1), preds.size(1), mask.size(1))
        preds_padded, targets_padded, mask_padded = [F.pad(tensor, (0, max_length - tensor.size(1)), value=0) for tensor in [preds, targets, mask]]

        useful_preds = preds_padded * mask_padded
        useful_targets = targets_padded * mask_padded
        
        # --- 余弦相似度计算 ---
        cos_sims = self.cos(useful_preds, useful_targets)
        loss_cos =  - cos_sims.mean()
        
        rounded_cos_sims = self.cos(useful_preds.round(), useful_targets.round())

        penalty_loss = self.penalty_loss(cos_sims)

        total_loss = loss_cos + 1*penalty_loss
        
        # --- 质量评分计算 ---
        cq_scores = torch.where(
            cos_sims < 0.7, 
            0.0,
            torch.where(cos_sims <=0.95, 100*(cos_sims-0.7)/0.25, 100.0)
        )

        rounded_cq_scores = torch.where(
            rounded_cos_sims < 0.7,
            0.0,
            torch.where(rounded_cos_sims <=0.95, 100*(rounded_cos_sims-0.7)/0.25, 100.0)
        )

        if torch.isnan(loss_cos).any():
            raise ValueError("NaN detected in loss_cos")
        return total_loss, cq_scores.mean(), rounded_cq_scores.mean()
    
    def penalty_loss(self, cos_sims):
        # --- 可微重参数化核心逻辑 ---
        # 1. 惩罚低分区（带安全边际）：cos_sim < (thr_low - margin) 时梯度放大
        penalty_threshold = self.thr_low - self.safety_margin
        penalty = F.relu(penalty_threshold - cos_sims)  # ReLU激活惩罚区
        penalty_loss = (penalty ** 2).mean()  # 二次惩罚
        
        # 2. 中间区线性奖励（权重自动控制）
        reward_zone = (cos_sims > self.thr_low) & (cos_sims < self.thr_high)
        reward =  (cos_sims[reward_zone] - self.thr_low)/(self.thr_high - self.thr_low)
        # 奖励项损失（负号转化：最大化奖励 = 最小化 -奖励）
        reward_loss = -reward.mean() if reward.numel() > 0 else 0.0
        # 3. 在计算reward后，对超阈值样本追加奖励
        super_reward = - F.relu(cos_sims - self.thr_high).mean()  # >0.95时激活
        # 4. 总损失（惩罚项+奖励项的组合）
        punishment_loss = ( 
            self.penalty_weight * penalty_loss +
            0.3 * reward_loss  # 奖励权重需小（防止梯度冲突）
            + 0.1 * super_reward # 超阈值样本的奖励
        )
        return punishment_loss

