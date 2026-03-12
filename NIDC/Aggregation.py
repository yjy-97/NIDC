import torch
import torch.nn.functional as F
import torch.nn as nn


class MGC(nn.Module):
    def __init__(self):

        super(MGC, self).__init__()

    def relative_position_encoding(self, M):
        pos = torch.arange(0, M).unsqueeze(0) - torch.arange(0, M).unsqueeze(1)
        relative_pos_encoding = torch.clamp(pos, min=0)  # 使用最大值来表示位置差异
        return relative_pos_encoding

    def max_relative_graph_convolution(self, X, relative_pos_encoding):
        B, M, D = X.shape
        adj_matrix = relative_pos_encoding.unsqueeze(0).expand(B, -1, -1)  # 扩展到batch维度
        adj_matrix = adj_matrix.float()
        aggregated_features = torch.matmul(adj_matrix.to('cuda:2') , X) # 聚合特征
        return F.relu(aggregated_features)  # 激活函数

    def forward(self, X):
        """
        前向传播函数，计算并执行相对位置编码和图卷积聚合
        :param X: 输入特征矩阵 (B, M, D)
        :return: 聚合后的特征矩阵 (B, M, D)
        """
        B, M, D = X.shape
        # 计算相对位置编码
        relative_pos_encoding = self.relative_position_encoding(M)

        # 执行最大相对图卷积
        output = self.max_relative_graph_convolution(X, relative_pos_encoding)

        return output


