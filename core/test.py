# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:29:37
# @Email:  cshzxie@gmail.com

import logging
import torch

import utils.helpers

from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.gr_implicitnet import GRImplicitNet
from utils.average_meter import AverageMeter
from utils.ImplicitMetrics import Metrics

from utils.ImplicitDataLoader import ImplicitDataset_inout

import open3d
import numpy as np



def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, imnet=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        if cfg.NETWORK.IMPLICIT_MODE == 1:
            test_dataset = ImplicitDataset_inout(cfg, phase='test')
        #elif cfg.NETWORK.IMPLICIT_MODE == 2:
        #    test_dataset = ImplicitDataset_onoff(cfg, phase='test')
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
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
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            partial_clouds = data['partial_cloud'].to(device='cuda')
            samples = data['samples'].to(device='cuda')
            labels = data['labels'].to(device='cuda')
            print(torch.sum(labels), len(labels[0][0]) )

            res, _loss = imnet(partial_clouds, samples, labels)  #获得计算数据 res:(B,1,N) samples:(B,N,3)
            test_losses.update([_loss.item()*1000])

            _metrics = Metrics.get(res, labels, samples)
            test_metrics.update(_metrics) #更新度量方法
            print("metrics: ", _metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)
            
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
                
            if cfg.b_save:
                import os
                path_save = "output/models/test/%s/%s/"%(taxonomy_id, model_id)
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
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
                print("test point cloud saved: %s"%(path_save + "predicted_%.2d.ply"%(model_idx%4)))

            #output reconstruction mesh
            if cfg.b_reconstruction:
                output_folder = "output/models/test/%s/%s/"%(taxonomy_id, model_id)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    
                    
                save_path = os.path.join(output_folder, 'predicted_%d.obj'%(model_idx))
                
                gen_mesh(cfg, imnet, cuda, data, save_path, use_octree=True)

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items: #打印度量名称
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics: #分类别打印误差值
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
        test_writer.add_scalar('Loss/Epoch/Loss', test_losses.avg(0), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
