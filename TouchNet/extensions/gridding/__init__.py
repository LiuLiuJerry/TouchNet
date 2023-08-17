# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-15 20:33:52
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-30 09:55:53
# @Email:  cshzxie@gmail.com

import torch

import gridding


class GriddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, ptcloud):
        #例：scale=64
        grid, grid_pt_weights, grid_pt_indexes = gridding.forward(-scale, scale - 1, -scale, scale - 1, -scale,
                                                                  scale - 1, ptcloud)
        # print(grid.size())             # torch.Size(batch_size, n_grid_vertices)
        # print(grid_pt_weights.size())  # torch.Size(batch_size, n_pts, 8, 3)
        # print(grid_pt_indexes.size())  # torch.Size(batch_size, n_pts, 8)
        ctx.save_for_backward(grid_pt_weights, grid_pt_indexes)

        return grid

    @staticmethod
    def backward(ctx, grad_grid):
        grid_pt_weights, grid_pt_indexes = ctx.saved_tensors
        grad_ptcloud = gridding.backward(grid_pt_weights, grid_pt_indexes, grad_grid)
        # print(grad_ptcloud.size())   # torch.Size(batch_size, n_pts, 3)

        return None, grad_ptcloud


class Gridding(torch.nn.Module):
    def __init__(self, scale=1):
        super(Gridding, self).__init__()
        self.scale = scale // 2

    def forward(self, ptcloud):
        ptcloud = ptcloud * self.scale #assuming origin shape range from [-1,1], transform the coordinate into [-scale, scale-1]
        _ptcloud = torch.split(ptcloud, 1, dim=0)
        grids = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            grids.append(GriddingFunction.apply(self.scale, p))

        return torch.cat(grids, dim=0).contiguous()


class GriddingReverseFunction(torch.autograd.Function):
    #根据网格特征生成点云
    @staticmethod
    def forward(ctx, scale, grid):
        ptcloud = gridding.rev_forward(scale, grid)
        ctx.save_for_backward(torch.Tensor([scale]), grid, ptcloud)
        return ptcloud
    #根据点云回传的梯度(grad_pycloud)算网格的梯度
    @staticmethod
    def backward(ctx, grad_ptcloud):
        scale, grid, ptcloud = ctx.saved_tensors
        scale = int(scale.item())
        grad_grid = gridding.rev_backward(ptcloud, grid, grad_ptcloud)
        grad_grid = grad_grid.view(-1, scale, scale, scale)
        return None, grad_grid


class GriddingReverse(torch.nn.Module):
    def __init__(self, scale=1):
        super(GriddingReverse, self).__init__()
        self.scale = scale

    def forward(self, grid):
        ptcloud = GriddingReverseFunction.apply(self.scale, grid)
        return ptcloud / self.scale * 2

#by Jerry
class GridSamplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, grid, ptcloud):
        pt_feature, grid_pt_weights, grid_pt_indexes = gridding.samp_forward(
            -scale, scale - 1, -scale, scale - 1, -scale, scale - 1, grid, ptcloud)
        ctx.save_for_backward(grid_pt_weights, grid_pt_indexes, ptcloud)
        return pt_feature

    @staticmethod
    def backward(ctx, grad_ptcloud):
        grid_pt_weights, grid_pt_indexes = ctx.saved_tensors
        grad_grid = gridding.samp_backward( grid_pt_weights, grid_pt_indexes, grad_ptcloud)
        return None, grad_grid
    
class GriddingSample(torch.nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, grid, ptcloud):
        # grid是每个网格顶点的特征
        pt_inout = GridSamplingFunction.apply(self.scale, grid, ptcloud)
