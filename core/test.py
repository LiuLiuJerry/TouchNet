# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:29:37
# @Email:  cshzxie@gmail.com

import logging
import torch

import utils.data_loaders
import utils.helpers

from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet import GRNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics

import open3d

#Jerry
from utils.loss import get_Geometric_Loss, VLossFlag


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None, b_save=False):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if grnet is None:
        grnet = GRNet(cfg)

        if torch.cuda.is_available():
            grnet = torch.nn.DataParallel(grnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS) #加载checkpoint
        grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
                                 alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)    # lgtm [py/unused-import]

    # Testing loop
    n_samples = len(test_data_loader)

    if cfg.v_flag ==VLossFlag.INITIAL_VERSION:
        test_losses = AverageMeter(['SparseLoss', 'DenseLoss'])
    elif cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
        test_losses = AverageMeter(['SparseLoss', 'DenseLoss', 'DensityLoss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)  #获得计算数据
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])

            #Jerry
            if cfg.v_flag == VLossFlag.INITIAL_VERSION:
                test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            elif cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
                _d_loss,_shape_loss, _density_loss = get_Geometric_Loss(sparse_ptcloud, data['gtcloud'])
                test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _density_loss*1000])


            _metrics = Metrics.get(dense_ptcloud, data['gtcloud'])
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if test_writer is not None and model_idx < 3:
                sparse_ptcloud = sparse_ptcloud.squeeze().cpu().numpy()
                sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
                test_writer.add_image('Model%02d/SparseReconstruction' % model_idx, sparse_ptcloud_img, epoch_idx, dataformats='HWC')
                dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
                dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud)
                test_writer.add_image('Model%02d/DenseReconstruction' % model_idx, dense_ptcloud_img, epoch_idx, dataformats='HWC')
                gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
                test_writer.add_image('Model%02d/GroundTruth' % model_idx, gt_ptcloud_img, epoch_idx, dataformats='HWC')
                #Jerry
                pt_ptcloud = data['partial_cloud'].squeeze().cpu().numpy()
                pt_ptcloud_img = utils.helpers.get_ptcloud_img(pt_ptcloud)
                test_writer.add_image('Model%02d/Input' % model_idx, pt_ptcloud_img, epoch_idx, dataformats='HWC')

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                            ], ['%.4f' % m for m in _metrics]))
            if b_save:
                import os
                path_save = "output/models/%s/%s/"%(taxonomy_id, model_id)
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
                utils.io.IO.put(path_save + "output.ply", dense_ptcloud.squeeze().cpu().numpy())
                utils.io.IO.put(path_save + "gt.ply", data['gtcloud'].squeeze().cpu().numpy())
                utils.io.IO.put(path_save + "partial.ply", data['partial_cloud'].squeeze().cpu().numpy())
                print("test point cloud saved: %s"%(path_save + "%.2d.ply"%(model_idx%4)))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics: #打印所有度量方式
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(1), epoch_idx)
        if cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
            test_writer.add_scalar('Loss/Epoch/Density', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
