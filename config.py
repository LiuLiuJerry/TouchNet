# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 17:18:04
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict
#Jerry
from utils.loss import VLossFlag

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
#Jerry: ShapeNetTouch config  --mug
__C.DATASETS.SHAPENETTOUCH                            = edict()
__C.DATASETS.SHAPENETTOUCH.N_RENDERINGS               = 4 #一个物体对应4个rendering
__C.DATASETS.SHAPENETTOUCH.N_POINTS                   = 2048 
# --参数1：subset（train/val）, 参数2：dc['taxonomy_id'], 物体类别的id  参数3： 数据id  参数4：i,同一个物体不同的exploration
__C.DATASETS.SHAPENETTOUCH.CATEGORY_FILE_PATH         = '/home/manager/data/ShapeCompletion/mug/ShapeNetTouch.json'
__C.DATASETS.SHAPENETTOUCH.PARTIAL_POINTS_PATH        = '/home/manager/data/ShapeCompletion/mug/%s/partial/%s/%s/path2048_%.2d.pcd'
__C.DATASETS.SHAPENETTOUCH.COMPLETE_POINTS_PATH       = '/home/manager/data/ShapeCompletion/mug/%s/complete/%s/%s/01.pcd'
__C.DATASETS.SHAPENETTOUCH.MESH_PATH                  = '/home/manager/data/ShapeCompletion/mug/%s/complete/%s/%s/o1_manifold_plus.obj' 
__C.DATASETS.SHAPENETTOUCH.SAMPLE_PATH_IN             = '/home/manager/data/ShapeCompletion/Airplane/%s/complete/%s/%s/sampled_point_in_%d.obj'
__C.DATASETS.SHAPENETTOUCH.SAMPLE_PATH_OUT            = '/home/manager/data/ShapeCompletion/Airplane/%s/complete/%s/%s/sampled_point_out_%d.obj'
__C.DATASETS.SHAPENETTOUCH.LOAD_MODE                  = 0 # 0:sample online   1:sample offline and load online. We have 20 iffline samples for Airplane and Car dataset
__C.DATASETS.SHAPENETTOUCH.N_SAMPLES                  = 20

#
# Dataset
#
__C.DATASET                                      = edict()
__C.DATASET.TRAIN_DATASET                        = 'ShapeNetTouch'
__C.DATASET.TEST_DATASET                         = 'ShapeNetTouch'

__C.v_flag = VLossFlag.INITIAL_VERSION
__C.b_save = False

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 4
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
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048 
__C.NETWORK.GRIDDING_LOSS_SCALES                 = [128]
__C.NETWORK.GRIDDING_LOSS_ALPHAS                 = [0.1]
__C.NETWORK.N_SAMPLING_MESHPOINTS                = 4000
__C.NETWORK.IMPLICIT_MODE                        = 1 # 1:in-out 2:on-off


#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 4 #32
__C.TRAIN.N_EPOCHS                               = 200
__C.TRAIN.SAVE_FREQ                              = 25
__C.TRAIN.LEARNING_RATE                          = 1e-4 #1e-4
__C.TRAIN.LR_MILESTONES                          = [50] # [15,50,80]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'F-Score_cloud_0.01'
