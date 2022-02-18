import json
import logging
import numpy as np
import random
import torch.utils.data.dataset.Dataset as Dataset
import torchvision.transforms as transforms
import trimesh

import utils.data_transforms

from enum import Enum, unique
from tqdm import tqdm

from utils.io import IO

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))

    return meshs

class ImplicitDataset(Dataset):
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.subset = phase
        self.is_train = phase == 'train'
        
        self.num_samplemesh_inout = CFG.NETWORK.N_SAMPLING_MESHPOINTS 
        
        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENETTOUCH.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read()) #从ShapeNet.json中读取物体目录（全部物体）

        #获取file_list
        self.data_list = self._get_data(self.cfg, self.subset)
        self.transforms = self._get_transforms(self.cfg)


    def _get_transforms(self, cfg, subset):
        transforms = []
        if subset is 'train':
            transforms.append({
                'callback':'RandomSamplePoints',
                'parameters':{
                    'callback':cfg.CONST.N_INPUT_POINTS
                },
                'objects':['partial_cloud']
            })
            transforms.append({
                'callback':'RandomMirrorPoints',
                'objects':[
                    'partial_cloud', 
                    'gtcloud'
                ]
            })
            transforms.append({
                'callback':'ToTensor',
                'objects': [
                    'partial_cloud', 
                    'gtcloud'
                ]})
        else:
            transforms.append({
                'callback':'RandomSamplePoints',
                'parameters':{
                    'callback':cfg.CONST.N_INPUT_POINTS
                },
                'objects':['partial_cloud']
                })
            transforms.append({
                'callback':'ToTensor',
                'objects': [
                    'partial_cloud', 
                    'gtcloud'
                ]})
        
        return utils.data_transforms.Compose(transforms)
    

    def _get_data(self, cfg, subset, n_renderings=1):
        #读取所有目标文件，并组织成字典的形式
        data_list = []
        
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dx['taxonomy_name']))
            samples = dc[subset]
            
            for s in tqdm(samples, leave=False): #遍历该类的全部文件
                
                gt_mesh_path = cfg.DATASETS.SHAPENETTOUCH.MESH_PATH % (subset, dc['taxonomy_id'], s)
                gt_mesh =  load_trimesh(gt_mesh_path)
                for i in range(n_renderings):
                    data_dc = {}
                    data_dc['texonomy_id'] = dc['taxonomy_id']
                    data_dc['model_id'] = s                    
                    
                    partial_path = cfg.DATASETS.SHAPENETTOUCH.PARTIAL_POINTS_PATH%(subset, dc['taxonomy_id'], s, i)
                    data_dc['partial_cloud'] = IO.get(data_dc['partial_cloud_path']).astype(np.float32)
                    data_dc['gt_mesh'] = gt_mesh
                    
                    data_list.append(data_dc)
            
        logging.info('Complete collecting files of the dataset. Total files:%d' % len(file_list))

        return data_list
    
    def select_sampling_method(self, mesh):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
            
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4*self.num_samplemesh_inout)
        sample_points = surface_points + np.random.normal(scale=self.cfg.SAMPLING_SIGMA, size=surface_points.shape)
        
        #add random points within generating space
        length = 1
        MIN = -0.5
        random_points = np.random.rand(self.num_samplemesh_inout // 4, 3)*length + MIN
        sample_points = np.concatenate([sample_points, random_points], axis=0)
        np.random.shuffle(sample_points)
        
        #label the points
        inside_idx = mesh.contains(sample_points)
        inside_points = sample_points[inside_idx]
        outside_points = sample_points[np.logical_not(inside_idx)]
        
        n_in = inside_points.shape[0]
        #限制点的个数
        n_s = self.num_samplemesh_inout // 2
        inside_points = inside_points[
            :n_s] if n_in > n_s else inside_points
        outside_points = outside_points[
            :n_s] if n_in > n_s else outside_points[:(self.num_samplemesh_inout-n_in)]
        
        samples = np.concatenate([inside_points, outside_points], 0)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])),
                                 np.ones((1,outside_points.shape[0]))], axis=1)
        
        #turn into torch tensor
        samples = torch.Tensor(samples).float()
        labels  = torch.Tensor(labels).float()
        
        return {
            'samples': samples,
            'labels': labels
        }
        
    def _get_options(self):
        self.options['required_item'] = ['partial_cloud', 'sampled_gt_points']


    def __len__(self):
        return len(self.data_list)

    def __get_item__(self, index):
        #读取需要获取的文件
        data_dc = self.data_list[index]
        data = {}
        for ri in self.options['required_items']:
            file_path = data_dc[ri]
            if ri is 'sampled_gt_points':#获取采样数据和label
                sample_data = self.select_sampling_method(data_dc['gt_mesh'])
                data[ri] = sample_data  #{samples, labels}
            else:
                data[ri] = data_dc[ri]
                
                
        if self.transforms is not None:
            data = self.transforms(data)

        return data_dc['taxomomy_id'], data_dc['model_id'], data