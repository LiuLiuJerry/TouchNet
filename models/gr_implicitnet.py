# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling

from models.SurfaceClassifier import SurfaceClassifier

from p_utils.common import gather_points
from p_utils.sampling import fps

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


class GRImplicitNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__() #调用父类
        self.opt = cfg
        
        self.surface_classifier = SurfaceClassifier(filter_channels=self.opt.mlp_dim,
                                                    no_residual=self.opt.no_residual,
                                                    last_op=nn.Sigmoid())
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
            #torch.nn.ReLU()
            torch.nn.LeakyReLU(0.2),
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            #torch.nn.ReLU()
            torch.nn.LeakyReLU(0.2),
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
            #torch.nn.LeakyReLU(0.2),
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
            #torch.nn.LeakyReLU(0.2),
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
            #torch.nn.LeakyReLU(0.2),
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
            #torch.nn.LeakyReLU(0.2),
        )
        
        self.error_term = nn.MSELoss()


        
    def filter(self, partial_cloud):
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
        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l  #直接把特征加进来？
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_r = self.dconv10(pt_features_32_r) 
             
        self.feat_list = [pt_features_64_r]
    
    def project(self, points):
        '''
        :parame points: [B,N,3]
        '''
        point_features = []
        uvw = points.unsqueeze(2) # [B, N, 1, 3]
        uvw = uvw.unsqueeze(2) # [B, N, 1, 1, 3]
        for f in self.feat_list:
            '''
            input: (N,C,D,H,W)
            grid : (N,D,H,W,3) 
            '''
            #grid_sample coordinate: [-1,1]
            p_f = torch.nn.functional.grid_sample(f, uvw, align_corners=True) #[B,C,N,1,1]
            point_features.append(p_f.view(p_f.shape[:3]))#[B,C,N]
        
        return point_features

    def project_add(self, points):
        '''
        :parame points: [B,N,3]
        '''
        #print(torch.min(points, axis=1), torch.max(points, axis=1))
        scale = 64
        scale_2 = scale * scale
        point_features = []
        points = points * scale//2 + scale//2
        #print(torch.min(points, axis=1), torch.max(points, axis=1))
        x = torch.floor(points[:,:,0]).type(torch.long)
        y = torch.floor(points[:,:,1]).type(torch.long)
        z = torch.floor(points[:,:,2]).type(torch.long)
        idx_0 = x * scale_2 + y * scale + z
        idx_1 = (x+1) * scale_2 + y * scale + z
        idx_2 = x * scale_2 + (y+1) * scale + z
        idx_3= (x+1) * scale_2 + (y+1) * scale + z
        idx_4 = x * scale_2 + y * scale + (z+1)
        idx_5 = (x+1) * scale_2 + y * scale + (z+1)
        idx_6 = x * scale_2 + (y+1) * scale + (z+1)
        idx_7 = (x+1) * scale_2 + (y+1) * scale + (z+1)

        #print(torch.min(idx_7, axis=0), torch.max(idx_7, axis=0))

        for f in self.feat_list:
            '''
            input: (N,C,D,H,W)
            grid : (N,D,H,W,3) 
            '''
            f = torch.permute(f, (0,2,3,4,1))
            f = torch.flatten(f, start_dim=1, end_dim=3)
            bt = f.shape[0]
            p_f = torch.zeros((bt,points.shape[1],f.shape[-1])).to(device='cuda')
            for i in range(bt):
                tmp1 = f[i,idx_0[i],:] + f[i,idx_1[i],:] + f[i,idx_2[i],:] + f[i,idx_3[i],:]
                tmp2 = f[i,idx_4[i],:] + f[i,idx_5[i],:] + f[i,idx_6[i],:] + f[i,idx_7[i],:] #[B,N,C]
                p_f[i,:] = tmp1 + tmp2
            point_features.append(torch.transpose(p_f, 1, 2))#[B,C,N]
        
        return point_features
      
    def query(self, points):
        #使用points对体素的特征进行查询
        point_features = self.project_add(points)  #投影到对应网路中
        point_features = point_features[-1]
        self.preds = self.surface_classifier(point_features)
        return self.preds
        
    def get_error(self, labels):
        return self.error_term(self.preds, labels)
    
    def get_preds(self):
        return self.preds

    def forward(self, partial_cloud, points, labels):
        # partial_cloud: B × num_sample_inout × 3
        # points: B × 2048 × 3
        self.filter(partial_cloud)
        #不进行反卷积，而是直接进行采样，通过mlp，获取inside/outside
        self.query(points)
        
        res = self.get_preds()
        
        error = self.get_error(labels)

        return res, error
