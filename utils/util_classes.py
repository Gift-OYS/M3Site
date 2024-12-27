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


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch, reduction="mean")
        self._update_centers(features, labels)

        return loss

    def _update_centers(self, features, labels):
        with torch.no_grad():
            unique_labels = labels.unique()
            for label in unique_labels:
                mask = labels == label
                selected_features = features[mask]
                if selected_features.size(0) > 0:
                    center_delta = selected_features.mean(dim=0) - self.centers[label]
                    self.centers[label] = self.centers[label] + self.alpha * center_delta


class InterClassLoss(nn.Module):
    def __init__(self, margin=0.0001):
        super(InterClassLoss, self).__init__()
        self.margin = margin

    def forward(self, centers):
        num_classes = centers.size(0)
        loss = 0
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                distance = torch.norm(centers[i] - centers[j])
                loss += torch.max(torch.tensor(0.0), self.margin - distance)
        return loss
