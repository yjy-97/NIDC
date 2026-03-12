import torch
import torch.nn as nn
import torch.nn.functional as F

class SCL(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=1.0, lambda_m=0.1, margin=1.0):
        super(SCL, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.lambda_m = lambda_m
        self.margin = margin
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)

        intra_loss = F.mse_loss(features, centers_batch.to('cuda:2'))

        center_distances = torch.cdist(self.centers, self.centers, p=2)

        margin_loss = torch.sum(F.relu(self.margin - center_distances))

        total_loss = self.lambda_c * intra_loss + self.lambda_m * margin_loss
        return total_loss



