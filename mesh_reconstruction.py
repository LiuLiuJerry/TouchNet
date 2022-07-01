#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Last Modified by:   jiarui liu
# @Last Modified time: 2022-04-18 19:28:24
# @Email:  18811758898@163.com

import logging
import os
import torch

from utils.ImplicitDataLoader import ImplicitDataset_inout
import utils.helpers
import utils.io

import numpy as np

from utils.mesh_util import reconstruction, save_obj_mesh


def gen_mesh(opt, net, cuda, data, save_path, use_octree=True):
    partial_clouds = data['partial_cloud'].to(device='cuda')

    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    net.filter(partial_clouds)

    b_min = np.array([-0.5, -0.5, -0.5])
    b_max = np.array([0.5, 0.5, 0.5])
    resolution = 256
    try:
        verts, faces, _, _ = reconstruction(
            net, cuda, resolution, b_min, b_max, use_octree=use_octree)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        save_obj_mesh(save_path, verts, faces)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')


