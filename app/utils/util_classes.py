import torch
import math
import torch.nn.functional as F
import torch.nn as nn


class MyPrint():
    def __init__(self, logger):
        self.logger = logger
        
    def pprint(self, *args):
        print(*args)
        log_message = ', '.join(str(arg) for arg in args)
        self.logger.info(log_message)


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = total_steps * warmup_ratio
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            cosine_step = self.last_epoch - self.warmup_steps
            cosine_steps = self.total_steps - self.warmup_steps
            return [
                base_lr * (1 + math.cos(math.pi * cosine_step / cosine_steps)) / 2
                for base_lr in self.base_lrs
            ]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # Focusing parameter
        self.alpha = alpha  # Balance parameter

    def forward(self, probs, targets, eps=1e-7):
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1))
        cross_entropy_loss = -torch.log(target_probs + eps)
        modulating_factor = (1 - target_probs) ** self.gamma
        focal_loss = self.alpha * modulating_factor * cross_entropy_loss
        return focal_loss.mean()


class BatchTripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric="euclidean"):
        """
        :param margin: Margin for triplet loss
        :param distance_metric: "euclidean" or "cosine"
        """
        super(BatchTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def _pairwise_distances(self, embeddings):
        """
        Compute pairwise distances between embeddings in the batch.
        """
        if self.distance_metric == "euclidean":
            # Squared pairwise Euclidean distances
            embeddings = F.normalize(embeddings, p=2, dim=1)
            dot_product = torch.matmul(embeddings, embeddings.t())
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
            distances = torch.clamp(distances, min=0.0)  # Avoid negative distances
            return torch.sqrt(distances + 1e-12)  # Add epsilon for numerical stability
        elif self.distance_metric == "cosine":
            # Cosine similarity -> distance
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            cosine_similarity = torch.matmul(normalized_embeddings, normalized_embeddings.t())
            distances = 1 - cosine_similarity  # Cosine distance
            return distances
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def forward(self, embeddings, labels):
        """
        Compute the triplet loss for a batch of embeddings and their corresponding labels.
        :param embeddings: Tensor of shape [batch_size, embedding_dim]
        :param labels: Tensor of shape [batch_size], integer class labels
        """
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)

        # Mask for valid triplets (Anchor-Positive and Anchor-Negative pairs)
        labels = labels.unsqueeze(1)
        is_positive = labels.eq(labels.t())  # Positive mask
        is_negative = ~is_positive  # Negative mask

        # For each anchor, find hardest positive and hardest negative
        anchor_positive_distances = distances * is_positive.float()  # Mask positive distances
        hardest_positive_distances = anchor_positive_distances.max(dim=1)[0]

        anchor_negative_distances = distances + 1e6 * is_positive.float()  # Mask negative distances
        hardest_negative_distances = anchor_negative_distances.min(dim=1)[0]

        # Compute Triplet Loss
        triplet_loss = F.relu(hardest_positive_distances - hardest_negative_distances + self.margin)
        return triplet_loss.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        """
        Center Loss 实现
        :param num_classes: 类别数
        :param feat_dim: 特征维度
        :param alpha: 中心点更新的学习率
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha

        # 初始化类别中心点 [num_classes, feat_dim]
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        """
        计算 Center Loss
        :param features: 输入特征 [batch_size, feat_dim]
        :param labels: 对应标签 [batch_size]
        :return: Center Loss
        """
        batch_size = features.size(0)
        
        # 获取每个样本的类别中心点 [batch_size, feat_dim]
        centers_batch = self.centers[labels]

        # 计算 Center Loss
        loss = F.mse_loss(features, centers_batch, reduction="mean")

        # 手动更新中心点
        self._update_centers(features, labels)

        return loss

    def _update_centers(self, features, labels):
        """
        更新类别中心点
        :param features: 当前 batch 的样本特征 [batch_size, feat_dim]
        :param labels: 当前 batch 的样本标签 [batch_size]
        """
        # 确保使用 torch.no_grad() 禁止梯度跟踪
        with torch.no_grad():
            unique_labels = labels.unique()  # 当前 batch 中的类别
            for label in unique_labels:
                mask = labels == label  # 筛选出该类别的样本
                selected_features = features[mask]  # 属于该类别的特征
                if selected_features.size(0) > 0:
                    # 计算中心点增量
                    center_delta = selected_features.mean(dim=0) - self.centers[label]
                    # 使用非 in-place 更新
                    self.centers[label] = self.centers[label] + self.alpha * center_delta


class InterClassLoss(nn.Module):
    def __init__(self, margin=0.0001):
        super(InterClassLoss, self).__init__()
        self.margin = margin

    def forward(self, centers):
        # 计算每对中心点之间的距离
        num_classes = centers.size(0)
        loss = 0
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                distance = torch.norm(centers[i] - centers[j])
                loss += torch.max(torch.tensor(0.0), self.margin - distance)
        return loss
