# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 17:18:04
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/home/SENSETIME/xiehaozhe/Datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/home/SENSETIME/xiehaozhe/Datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8 #一个物体对应8个rendering？
__C.DATASETS.SHAPENET.N_POINTS                   = 16384
# 参数1：subset（train/val）, 参数2：dc['taxonomy_id'],物体类别的id  参数3：s,数据id  参数：i，可能是同一个物体不同的partial？)
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/partial/%s/%s/%.2d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.KITTI                               = edict()
__C.DATASETS.KITTI.CATEGORY_FILE_PATH            = './datasets/KITTI.json'
__C.DATASETS.KITTI.PARTIAL_POINTS_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/KITTI/cars/%s.pcd'
__C.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/KITTI/bboxes/%s.txt'

#Jerry: ShapeNetTouch config
__C.DATASETS.SHAPENETTOUCH                           = edict()
__C.DATASETS.SHAPENETTOUCH.CATEGORY_FILE_PATH         = './datasets/ShapeNetTouch.json'
__C.DATASETS.SHAPENETTOUCH.N_RENDERINGS        = 4 #一个物体对应8个rendering？
__C.DATASETS.SHAPENETTOUCH.N_POINTS                   = 2048 #好像也没用到？
# 参数1：subset（train/val）, 参数2：dc['taxonomy_id'],物体类别的id  参数3：s,数据id  参数：i，可能是同一个物体不同的partial？)
__C.DATASETS.SHAPENETTOUCH.PARTIAL_POINTS_PATH        = '/home/jerry/data/ShapeNetCompletion/%s/partial/%s/%s/%.2d.pcd'
__C.DATASETS.SHAPENETTOUCH.COMPLETE_POINTS_PATH       = '/home/jerry/data/ShapeNetCompletion/%s/complete/%s/%s/01.pcd'

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTI
#__C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
#__C.DATASET.TEST_DATASET                         = 'ShapeNet'
__C.DATASET.TRAIN_DATASET                        = 'ShapeNetTouch'
__C.DATASET.TEST_DATASET                         = 'ShapeNetTouch'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048   #目标输入的点云点的数目

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                     = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048  #没有用到
__C.NETWORK.GRIDDING_LOSS_SCALES                 = [128]
__C.NETWORK.GRIDDING_LOSS_ALPHAS                 = [0.1]

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 4 #32
__C.TRAIN.N_EPOCHS                               = 50#150
__C.TRAIN.SAVE_FREQ                              = 25
__C.TRAIN.LEARNING_RATE                          = 1e-4
__C.TRAIN.LR_MILESTONES                          = [50]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
