#! /usr/bin/python3
# @Author: Jiarui Liu
# @Date:   2022-09-28 22:37
# @Last Modified by:   Jiarui Liu
# @Last Modified time: 2022-09-28 22:37
# @Email:  18811758898@163.com

import argparse
import cv2
import logging
import matplotlib
import os
import open3d    # lgtm [py/unused-import]
import sys
import torch    # lgtm [py/unused-import]
# Fix no $DISPLAY environment variable
matplotlib.use('Agg')
# Fix deadlock in DataLoader
cv2.setNumThreads(0)

from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net
from core.inference import inference_net

#Jerry
from utils.loss import VLossFlag

import warnings
warnings.filterwarnings("ignore")

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of R2Net runner')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to use', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true', default=True)
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true', default=False)
    #parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None) 
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=
                        '/home/manager/codes/TouchCompletion/shape-repair/output/ImplicitNet-Car-ckpt-best.pth') 
    parser.add_argument('--save', dest="save", help='save results during test', default=True) 
    parser.add_argument('--mlp_dim', dest="mlp_dim", default=[16, 64, 128, 32, 1], type=int)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

    cfg.b_save = args.save
    cfg.b_reconstruction = 0
    #network parameters
    cfg.mlp_dim = args.mlp_dim
    cfg.no_residual = True
    cfg.SAMPLING_SIGMA = 0.1

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test and not args.inference:
        train_net(cfg) #进行训练
    else:
        if 'WEIGHTS' not in cfg.CONST or not os.path.exists(cfg.CONST.WEIGHTS):
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)

        if args.test:
            test_net(cfg)
        else:
            inference_net(cfg)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on https://github.com/hzxie/GRNet")

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)

    main()
