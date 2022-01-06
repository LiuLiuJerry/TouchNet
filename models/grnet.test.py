# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Jerry Liu
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  jiarui.liu@ia.ac.cn

import torch

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1) #batch_size * (262144+2048) * 3

        _ptcloud = torch.split(pred_cloud, 1, dim=0) #以batch为单位分开，用于迭代
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0) #过滤掉全是0的点
            n_pts = p.size(1)
            if n_pts < self.n_points:  #随机取N_INPUT_POINTS（2048）个点
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()  #重新拼接到同一个张量中


class GRNet(torch.nn.Module):
    def __init__(self, cfg):
        super(GRNet, self).__init__()
        self.gridding = Gridding(scale=64)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(  #两层全链接
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256*2, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128*2, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64*2, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32*2, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.fc_conv_11 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=1, stride=1, bias=False, padding=0),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=64)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling() #根据点云 对特征进行采样
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(448, 112),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(112, 24)

    def forward(self, data):
        partial_cloud = data['partial_cloud']
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        # print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(pt_features_64_l)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        # print(features.size())          # torch.Size([batch_size, 2048])
        '''pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l  #直接把特征加进来？
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_r = self.dconv10(pt_features_32_r) + pt_features_64_l'''

        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4)  #特征拼接起来？
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(torch.cat((pt_features_4_r, pt_features_4_l), dim=1))
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(torch.cat((pt_features_8_r, pt_features_8_l), dim=1))
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r =  self.dconv9(torch.cat((pt_features_16_r, pt_features_16_l), dim=1))
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_2r =  self.dconv10(torch.cat((pt_features_32_r, pt_features_32_l), dim=1))  

        #debug
        #print("min pt_features_64_2r", torch.min(pt_features_64_2r))
        #print("max pt_features_64_2r", torch.max(pt_features_64_2r))
        pt_features_64_4 = self.fc_conv_11(torch.cat((pt_features_64_2r, pt_features_64_l), dim=1))

        #debug
        #print("min pt_features_64_4", torch.min(pt_features_64_4))
        #print("max pt_features_64_4", torch.max(pt_features_64_4))

        # print(pt_features_64_r.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev( pt_features_64_4.squeeze(dim=1)) ####每个网格产生一个点云的点， 共64*64*64个
        # print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)   ####采样到2048
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])   ####后续目标为 为每个sparse cloud中的点学习一个offset
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, 2048, 256)
        # print(point_features_32.size()) # torch.Size([batch_size, 2048, 256])
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, 2048, 512)
        # print(point_features_16.size()) # torch.Size([batch_size, 2048, 512])
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, 2048, 1024)
        # print(point_features_8.size())  # torch.Size([batch_size, 2048, 1024])
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 448])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 112])
        point_offset = self.fc14(point_features).view(-1, 16384, 3)  #对训练得到的稀疏点云的每个点学习一个offset，然后加到稀疏点云上，实现一个点变8个点
        # print(point_features.size())    # torch.Size([batch_size, 16384, 3])
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, 16384, 3) + point_offset
        # print(dense_cloud.size())       # torch.Size([batch_size, 16384, 3])

        return sparse_cloud, dense_cloud
