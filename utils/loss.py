import numpy as np
import torch
from enum import Enum

class VLossFlag(Enum):
     INITIAL_VERSION = 1
     DENSITY_LOSS_VERSION = 2

def get_Geometric_Loss(predictedPts, targetpoints, nnk=8, densityWeight=1):

    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = torch.sqrt( square_dist ) # 开方
    minRow = torch.min(dist, axis=2).values ## 在降维后的第二维，即y那维找最小
    minCol = torch.min(dist, axis=1).values## 在x那一维找最小值
    shapeLoss = torch.mean(minRow) + torch.mean(minCol) ## 在[batchsize,x]取平均

    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    dist2 = torch.sqrt(square_dist2)
    knndis = torch.topk(torch.negative(dist), k=nnk) #返回pred到gt距离最小的8个数
    knndis2 = torch.topk(torch.negative(dist2), k=nnk) #返回gt到gt距离最小的8个数
    densityLoss = torch.mean(torch.abs(knndis.values - knndis2.values))

    data_loss = shapeLoss + densityLoss * densityWeight
    return data_loss, shapeLoss, densityLoss


def pairwise_l2_norm2_batch(x, y, scope=None):
        nump_x = x.shape[1] #point number of each shape
        nump_y = y.shape[1]

        xx = torch.unsqueeze(x, -1)
        ### stack:矩阵拼接  ### tile:某维度重复，如下列代码表示将最后一维重复nump_y次
        xx = torch.tile(xx, [1, 1, 1, nump_y]) 

        yy = torch.unsqueeze(y, -1) # [batch_size, npts, 3]
        yy = torch.tile(yy, [1, 1, 1, nump_x])
        
        yy = torch.transpose(yy, dim0=1, dim1=3) # 交换张量的不同维度相当于1和3维的转置

        diff = torch.subtract(xx, yy) # 做差，xx中每个点和yy中每个点的差
        square_diff = torch.square(diff) # 平方

        square_dist = torch.sum(square_diff, 2) # 平方和，特征维求平方和，降维

        return square_dist    