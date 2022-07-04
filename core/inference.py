# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-23 11:46:33
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:12:44
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

from utils.ImplicitDataLoader import ImplicitDataset_inout
import utils.helpers
import utils.io

from models.gr_implicitnet import GRImplicitNet
from mesh_reconstruction import gen_mesh


def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if cfg.NETWORK.IMPLICIT_MODE == 1:
        test_dataset = ImplicitDataset_inout(cfg, phase='val')
    #elif cfg.NETWORK.IMPLICIT_MODE == 2:
    #    test_dataset = ImplicitDataset_onoff(cfg, phase='val')

    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=1,
                                                       num_workers=4, #cfg.CONST.NUM_WORKERS,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    imnet = GRImplicitNet(cfg)
    # set cuda
    cuda = torch.device('cuda:%d' % 0) if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        imnet = torch.nn.DataParallel(imnet).cuda()

    # Load the pretrained model from a checkpoint
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    imnet.load_state_dict(checkpoint['imnet'])

    # Switch models to evaluation mode
    imnet.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)


            '''
            partial_clouds = data['partial_cloud'].to(device='cuda')
            samples = data['samples'].to(device='cuda')
            labels = data['labels'].to(device='cuda')
                    
            res, _loss = imnet(partial_clouds, samples, labels)  #获得计算数据 res:(B,1,N) samples:(B,N,3)
            
            output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            output_file_path = os.path.join(output_folder, '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, dense_ptcloud.squeeze().cpu().numpy())

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, output_file_path))'''
            
            output_folder = "output/models/val/%s/%s/"%(taxonomy_id, model_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
                
            save_path = os.path.join(output_folder, 'reconstructed_%d.obj'%(model_idx))
            
            gen_mesh(cfg, imnet, cuda, data, save_path, use_octree=True)
