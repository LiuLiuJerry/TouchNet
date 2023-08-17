# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 18:34:19
# @Email:  cshzxie@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.mplot3d import Axes3D


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
       type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    #ax.axis('scaled')
    ax.view_init(30, 45)

    #max, min = np.max(ptcloud), np.min(ptcloud)
    p_max = 0.6
    p_min = -0.6
    ax.set_xbound(p_min, p_max)
    ax.set_ybound(p_min, p_max)
    ax.set_zbound(p_min, p_max)
    
    dx = 1.2
    xx = [p_min, p_max, p_max, p_min, p_min]
    yy = [p_max, p_max, p_min, p_min, p_max]
    kwargs1 = {'linewidth':1, 'color':'black', 'linestyle':'-'}
    kwargs2 = {'linewidth':1, 'color':'black', 'linestyle':'--'}
    ax.plot(xx, yy, p_max, **kwargs1)
    ax.plot(xx[:3], yy[:3], p_min, **kwargs1)
    ax.plot(xx[2:], yy[2:], p_min, **kwargs2)
    for n in range(3):
        ax.plot([xx[n], xx[n]], [yy[n], yy[n]], [p_min, p_max], **kwargs1)
    ax.plot([xx[3], xx[3]], [yy[3], yy[3]], [p_min, p_max], **kwargs2)
    
   
    
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img
