# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-04 11:01:37
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.data_loaders
import utils.helpers

from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from core.test import test_net
from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet import GRNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics

#Jerry
from utils.loss import get_Geometric_Loss, VLossFlag

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    #读取点云
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  collate_fn=utils.data_loaders.collate_fn,
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
    grnet = GRNet(cfg)
    grnet.apply(utils.helpers.init_weights)
    logging.debug('Parameters in GRNet: %d.' % utils.helpers.count_parameters(grnet))


    # Move the network to GPU if possible
    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)
    #grnet_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(grnet_optimizer, gamma=0.95)

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        grnet.load_state_dict(checkpoint['grnet'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        if cfg.v_flag ==VLossFlag.INITIAL_VERSION:
            losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        elif cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
            losses = AverageMeter(['SparseLoss', 'DenseLoss', 'DensityLoss'])

        grnet.train()

        batch_end_time = time()
        n_batches = len(train_data_loader) #每个epoch都将所有数据分为不同batch
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader): #每次取一个batch
            data_time.update(time() - batch_end_time)
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            import sys
            b_debug = True if sys.gettrace() else False
            if b_debug:
                for  name , params in grnet.named_parameters():
                    #print('-->name:', name, '-->grad_requirs:', params.requires_grad, '-->grad_value:', params.grad)
                    print('-->name:', name, '-->grad_requirs:', params.requires_grad, '-->value:', params.values)


            sparse_ptcloud, dense_ptcloud = grnet(data)
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])  #稀疏点云和稠密点云一起算损失
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            
            #Jerry
            if cfg.v_flag == VLossFlag.INITIAL_VERSION:
                _loss = sparse_loss  + dense_loss  #
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            elif cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
                _d_loss,_shape_loss, _density_loss = get_Geometric_Loss(dense_ptcloud, data['gtcloud'])
                _loss = sparse_loss + dense_loss + _density_loss
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, _density_loss*1000])

            grnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()


            n_itr = (epoch_idx - 1) * n_batches + batch_idx  #迭代次数
            train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
            if cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
                train_writer.add_scalar("Loss/Batch/Density", _density_loss.item()*1000, n_itr)


            batch_time.update(time() - batch_end_time)
            batch_end_time = time()


            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))
        
            if cfg.b_save and epoch_idx == cfg.TRAIN.N_EPOCHS :
                for i in range(cfg.TRAIN.BATCH_SIZE):
                    path_save = "output/models/train/%s/%s/"%(taxonomy_ids[i], model_ids[i])
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    utils.io.IO.put(path_save + "sparse_%d.ply"%(i), sparse_ptcloud.cpu().detach().numpy()[i])
                    utils.io.IO.put(path_save + "dense_%d.ply"%(i), dense_ptcloud.cpu().detach().numpy()[i])
                    utils.io.IO.put(path_save + "gt.ply", data['gtcloud'].cpu().detach().numpy()[i])
                    utils.io.IO.put(path_save + "partial_%d.ply"%(i), data['partial_cloud'].cpu().detach().numpy()[i])
                    print("train point cloud saved: %s"%(path_save + "ep%.2d_%.2d.ply"%(epoch_idx, i)))


        grnet_lr_scheduler.step() #更新学习率
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch_idx)
        #Jerry
        if cfg.v_flag == VLossFlag.DENSITY_LOSS_VERSION:
            train_writer.add_scalar('Loss/Epoch/Density', losses.avg(2), epoch_idx)
        #train_writer.add_scalar('Batchnorm/conv1_mean', grnet.module.conv1._modules["1"].running_mean[0], epoch_idx)
        #train_writer.add_scalar('Batchnorm/conv1_var', grnet.module.conv1._modules["1"].running_var[0], epoch_idx)

        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        # Validate the current model 每个epoch验证一次
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, grnet)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'grnet': grnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics


    train_writer.close()
    val_writer.close()
