# -*- coding: utf-8 -*-
# @Author: Jiarui Liu
# @Date:   2022-09-28 22:37
# @Last Modified by:   Jiarui Liu
# @Last Modified time: 2022-09-28 22:37
# @Email:  18811758898@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, no_residual=True, last_op=None):
        super().__init__()
        self.filters = []
        self.no_residual = no_residual
        self.last_op = last_op
        
        if self.no_residual:
            for l in range(0, len(filter_channels)-1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1
                ))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter)-1):
                if l != 0:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1
                        )
                    )
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1
                    ))
                self.add_module('conv%d' % l, self.filters[l])
                
    def forward(self, feature):
        '''
        :param feature: list of [B,C,N] tensors
        :return : [B, N]
        '''
        y = feature
        tmpy = feature
        for i , f in enumerate(self.filters):
            if self.no_residual:
                y = self.filters[i](y)
            else:
                if i == 0:
                    y = self.filters[i](y)
                else:
                    y = self.filters[i](torch.cat([y, tmpy], 1))
                    
            #激活函数
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
                
            if self.last_op:
                y = self.last_op(y)
                
        return y
        
        