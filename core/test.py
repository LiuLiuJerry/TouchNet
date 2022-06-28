# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:29:37
# @Email:  cshzxie@gmail.com

import logging
import torch

<<<<<<< HEAD
=======
import utils.data_loaders
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
import utils.helpers

from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
<<<<<<< HEAD
from models.gr_implicitnet import GRImplicitNet
from utils.average_meter import AverageMeter
from utils.ImplicitMetrics import Metrics

from utils.ImplicitDataLoader import ImplicitDataset_inout, ImplicitDataset_onoff

import open3d
import numpy as np



def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, imnet=None):
=======
from models.grnet import GRNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics

import open3d

#Jerry
from utils.loss import get_Geometric_Loss, VLossFlag


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None):
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
<<<<<<< HEAD
        if cfg.NETWORK.IMPLICIT_MODE == 1:
            test_dataset = ImplicitDataset_inout(cfg, phase='test')
        elif cfg.NETWORK.IMPLICIT_MODE == 2:
            test_dataset = ImplicitDataset_onoff(cfg, phase='test')
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
=======
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
<<<<<<< HEAD
    if imnet is None:
        imnet = GRImplicitNet(cfg)

        if torch.cuda.is_available():
            imnet = torch.nn.DataParallel(imnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS) #加载checkpoint
        imnet.load_state_dict(checkpoint['imnet'])

    # Switch models to evaluation mode
    imnet.eval() #禁用BN和dropout
    # set cuda
    cuda = torch.device('cuda:%d' % 0) if torch.cuda.is_available() else torch.device('cpu')
    
    
    # Testing loop
    n_samples = len(test_data_loader)
    
    test_losses = AverageMeter((['Loss']))

    # evaluation metrics
=======
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
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
<<<<<<< HEAD
            partial_clouds = data['partial_cloud'].to(device='cuda')
            samples = data['samples'].to(device='cuda')
            labels = data['labels'].to(device='cuda')
            print(torch.sum(labels), len(labels[0][0]) )

            res, _loss = imnet(partial_clouds, samples, labels)  #获得计算数据 res:(B,1,N) samples:(B,N,3)
            test_losses.update([_loss.item()*1000])

            _metrics = Metrics.get(res, labels, samples)
            test_metrics.update(_metrics) #更新度量方法
            print("metrics: ", _metrics)
=======
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)  #获得计算数据
            #sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            
            sparse_loss = gridding_loss(sparse_ptcloud, data['gtcloud'])+chamfer_dist(sparse_ptcloud, data['gtcloud'])
            #dense_loss = gridding_loss(dense_ptcloud, data['gtcloud'])

            #Jerry
            if cfg.v_flag == VLossFlag.INITIAL_VERSION:
                test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            elif cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
                _d_loss,_shape_loss, _density_loss = get_Geometric_Loss(dense_ptcloud, data['gtcloud'])
                test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _density_loss*1000])


            _metrics = Metrics.get(dense_ptcloud, data['gtcloud'])
            test_metrics.update(_metrics)
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)
<<<<<<< HEAD
            
            #判断是否每次都预测得到相同值
            if model_idx > 0:
                y_pre_cpu = y_cpu.copy()
                y_cpu = res.squeeze(1).detach().cpu().numpy()
                y_same = np.argwhere(y_cpu == y_pre_cpu) #输出比较少说明采样点基本不一致，是对的
                y_in = np.argwhere(y_cpu>0.5)
                #print(y_same)

            #前3个模型渲染成图，作为tensorboard的展示
            if test_writer is not None and model_idx < 3:
                y_cpu = res.squeeze(1).cpu().numpy()
                samples_cpu = samples.cpu().numpy()
                ptcloud_cpu = samples_cpu[y_cpu > 0.5]
                ptcloud_img = utils.helpers.get_ptcloud_img(ptcloud_cpu)
                test_writer.add_image('Model%02d/SparseReconstruction' % model_idx, ptcloud_img, epoch_idx, dataformats='HWC')
                partial_img = utils.helpers.get_ptcloud_img(data['partial_cloud'][0])
                test_writer.add_image('Model%02d/TouchExploration' % model_idx, partial_img, epoch_idx, dataformats='HWC')
                labels_cpu = labels.squeeze(1).cpu().numpy()
                gtcloud_cpu = samples_cpu[labels_cpu>0.5]
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gtcloud_cpu)
                test_writer.add_image('Model%02d/GroundTruth' % model_idx, gt_ptcloud_img, epoch_idx, dataformats='HWC')


            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                           ], ['%.4f' % m for m in _metrics]))
            if(_metrics[1] > 5):
                print("wrong metric! ", _metrics[1])
                
=======

            if test_writer is not None and model_idx < 3:
                sparse_ptcloud_cpu = sparse_ptcloud.squeeze().cpu().numpy()
                sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud_cpu)
                test_writer.add_image('Model%02d/SparseReconstruction' % model_idx, sparse_ptcloud_img, epoch_idx, dataformats='HWC')
                dense_ptcloud_cpu = dense_ptcloud.squeeze().cpu().numpy()
                dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud_cpu)
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
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
            if cfg.b_save:
                import os
                path_save = "output/models/test/%s/%s/"%(taxonomy_id, model_id)
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
<<<<<<< HEAD
                # predicted labels
                y_cpu = res.squeeze(1).cpu().numpy()
                samples_cpu = samples.cpu().numpy()
                ptcloud_cpu = samples_cpu[y_cpu > 0.5]
                # ground truth labels
                labels_cpu = labels.squeeze(1).cpu().numpy()
                gtcloud_cpu = samples_cpu[labels_cpu > 0.5]
                
                utils.io.IO.put(path_save + "predicted_%.2d.ply"%(model_idx%4), ptcloud_cpu)
                utils.io.IO.put(path_save + "gt.ply", gtcloud_cpu)
                utils.io.IO.put(path_save + "partial_%.2d.ply"%(model_idx%4), data['partial_cloud'].cpu().numpy()[0])
                print("test point cloud saved: %s"%(path_save + "%.2d.ply"%(model_idx%4)))

            #output reconstruction mesh
            if cfg.b_reconstruction:
                output_folder = "output/models/test/%s/%s/"%(taxonomy_id, model_id)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    
                    
                save_path = os.path.join(output_folder, 'predicted_%d.obj'%(model_idx))
                
                gen_mesh(cfg, imnet, cuda, data, save_path, use_octree=True)
=======
                utils.io.IO.put(path_save + "sparse.ply", sparse_ptcloud.cpu().detach().numpy()[0])
                utils.io.IO.put(path_save + "dense.ply", dense_ptcloud.cpu().detach().numpy()[0])
                utils.io.IO.put(path_save + "gt.ply", data['gtcloud'].cpu().numpy()[0])
                utils.io.IO.put(path_save + "partial.ply", data['partial_cloud'].cpu().numpy()[0])
                print("test point cloud saved: %s"%(path_save + "%.2d.ply"%(model_idx%4)))

>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
<<<<<<< HEAD
    for metric in test_metrics.items: #打印度量名称
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics: #分类别打印误差值
=======
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics: #打印所有度量方式
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
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
<<<<<<< HEAD
        test_writer.add_scalar('Loss/Epoch/Loss', test_losses.avg(0), epoch_idx)
=======
        test_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(1), epoch_idx)
        if cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
            test_writer.add_scalar('Loss/Epoch/Density', test_losses.avg(2), epoch_idx)
>>>>>>> d797c9a3b87a78c76f852e04ce9809d4580e0d71
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
