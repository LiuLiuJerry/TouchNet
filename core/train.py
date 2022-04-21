# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-04 11:01:37
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.helpers

from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from core.test import test_net
from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from utils.average_meter import AverageMeter
from utils.metrics import Metrics

from models.gr_implicitnet import GRImplicitNet
from utils.ImplicitDataLoader import ImplicitDataset_inout, ImplicitDataset_onoff

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if cfg.NETWORK.IMPLICIT_MODE == 1:
        train_dataset = ImplicitDataset_inout(cfg, phase='train')
        test_dataset = ImplicitDataset_inout(cfg, phase='test')
    elif cfg.NETWORK.IMPLICIT_MODE == 2:
        train_dataset = ImplicitDataset_onff(cfg, phase='train')
        test_dataset = ImplicitDataset_onff(cfg, phase='test')
        
    #读取点云
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    imnet = GRImplicitNet(cfg)
    imnet.apply(utils.helpers.init_weights)
    logging.debug('Parameters in GRImplicitNet: %d.' % utils.helpers.count_parameters(imnet))


    # Move the network to GPU if possible
    if torch.cuda.is_available():
        imnet = torch.nn.DataParallel(imnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, imnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)
    #grnet_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(grnet_optimizer, gamma=0.95)

    

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        imnet.load_state_dict(checkpoint['imnet'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        imnet.train() #开启BN和dropout

        batch_end_time = time()
        n_batches = len(train_data_loader) #每个epoch都将所有数据分为不同batch
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader): #每次取一个batch
 
            data_time.update(time() - batch_end_time)

            partial_clouds = data['partial_cloud'].to(device='cuda')
            samples = data['samples'].to(device='cuda')
            labels = data['labels'].to(device='cuda')

            import sys
            b_debug = True if sys.gettrace() else False
            if b_debug:
                for  name , params in imnet.named_parameters():
                    #print('-->name:', name, '-->grad_requirs:', params.requires_grad, '-->grad_value:', params.grad)
                    print('-->name:', name, '-->grad_requirs:', params.requires_grad, '-->value:', params.values)


            res, _loss = imnet(partial_clouds, samples, labels)
            losses.update([_loss.item()*1000])


            imnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()


            n_itr = (epoch_idx - 1) * n_batches + batch_idx  #迭代次数
            train_writer.add_scalar('Loss/Batch/L2_loss', _loss.item() * 1000, n_itr)


            batch_time.update(time() - batch_end_time)
            batch_end_time = time()


            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))
        
            if cfg.b_save and epoch_idx == cfg.TRAIN.N_EPOCHS :

                # predicted labels
                y_cpu = res.squeeze(1).detach().cpu().numpy()
                samples_cpu = samples.cpu().numpy()
                
                # ground truth labels
                labels_cpu = labels.squeeze(1).cpu().numpy()
                
                for i in range(cfg.TRAIN.BATCH_SIZE):
                    path_save = "output/models/train/%s/%s/"%(taxonomy_ids[i], model_ids[i])
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)

                    ptcloud_cpu = samples_cpu[0][y_cpu[0] > 0.5]
                    gtcloud_cpu = samples_cpu[0][labels_cpu[0] > 0.5]
                    
                    utils.io.IO.put(path_save + "gt.ply", gtcloud_cpu)
                    utils.io.IO.put(path_save + "partial.ply", data['partial_cloud'].cpu().numpy()[0])
                    utils.io.IO.put(path_save + "predicted_%d.ply"%(i), ptcloud_cpu)
                
                    print("train point cloud saved: %s"%(path_save + "ep%.2d_%.2d.ply"%(epoch_idx, i)))


        grnet_lr_scheduler.step() #更新学习率
        epoch_end_time = time()
        
        #更新tensorboard
        train_writer.add_scalar('Loss/Epoch/L2_loss', losses.avg(0), epoch_idx)
        
        #打印信息
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        # Validate the current model 每个epoch验证一次
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, imnet)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'imnet': imnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics


    train_writer.close()
    val_writer.close()
