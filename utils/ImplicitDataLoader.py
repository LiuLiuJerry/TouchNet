import json
import logging
import numpy as np
import random
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import trimesh
import os

import utils.data_transforms

from enum import Enum, unique
from tqdm import tqdm

from utils.io import IO

def load_trimesh(root_dir):
    #folders = os.listdir(root_dir)
    #meshs = {}
    #for i, f in enumerate(folders):
    #    sub_name = f
    #    meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))
    mesh = trimesh.load(root_dir, force='mesh', process=False)
    '''if not mesh.is_watertight:
        face_idx = trimesh.repair.broken_faces(mesh)
        trimesh.repair.fill_holes(mesh)
        print('hole filled')'''
    print("number of faces: %d"%(len(mesh.triangles)))
        
    return mesh

def load_gt(sample_path_in, sample_path_out):
    mesh_in = trimesh.load(sample_path_in, force='mesh', process=False)
    mesh_out = trimesh.load(sample_path_out, force='mesh', process=False)
    points = np.concatenate([mesh_in.vertices, mesh_out.vertices], axis = 0).astype(np.float32)

    
    return points, len(mesh_in.vertices), len(mesh_out.vertices)

#in this version the dataset is represented as 0-inside, 1-outside
class ImplicitDataset_inout(torch.utils.data.dataset.Dataset):
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.subset = phase
        self.is_train = phase == 'train'
        
        self.num_samplemesh_inout = cfg.NETWORK.N_SAMPLING_MESHPOINTS 
        self.options = self._get_options()
        
        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENETTOUCH.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read()) #从ShapeNet.json中读取物体目录（全部物体）

        #获取file_list
        self.data_list = self._get_data(self.cfg, self.subset, n_renderings=self.cfg.DATASETS.SHAPENETTOUCH.N_RENDERINGS)
        self.transforms = self._get_transforms(self.cfg, subset=phase)


    def _get_transforms(self, cfg, subset):
        transforms = []
        if subset == 'train':
            transforms.append({
                'callback':'RandomSamplePoints',
                'parameters':{
                    'n_points':cfg.CONST.N_INPUT_POINTS
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
                    'n_points':cfg.CONST.N_INPUT_POINTS
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
        file_len = 0
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]#[:20]
            
            
            file_len = file_len + len(samples)
            for s in tqdm(samples, leave=False): #遍历该类的全部文件
                
                gt_mesh_path = cfg.DATASETS.SHAPENETTOUCH.MESH_PATH % (subset, dc['taxonomy_id'], s)


                data_dc = {}
                data_dc['taxonomy_id'] = dc['taxonomy_id']
                data_dc['model_id'] = s
                data_dc['partial_cloud'] = []
                data_dc['samples_path_in'] = []
                data_dc['samples_path_out'] = []
                if self.cfg. DATASETS.SHAPENETTOUCH.LOAD_MODE == 0:
                    gt_mesh =  load_trimesh(gt_mesh_path)
                    data_dc['gt_mesh'] = gt_mesh
                for i in range(n_renderings):

                    partial_path = cfg.DATASETS.SHAPENETTOUCH.PARTIAL_POINTS_PATH%(subset, dc['taxonomy_id'], s, i)
                    partial_tmp = IO.get(partial_path).astype(np.float32)
                    data_dc['partial_cloud'].append(partial_tmp)
                   
                    
                    if self.cfg.DATASETS.SHAPENETTOUCH.LOAD_MODE == 1:
                        sample_path_in = cfg.DATASETS.SHAPENETTOUCH.SAMPLE_PATH_IN%(subset, dc['taxonomy_id'], s, i)
                        sample_path_out = cfg.DATASETS.SHAPENETTOUCH.SAMPLE_PATH_OUT%(subset, dc['taxonomy_id'], s, i)


                        data_dc['samples_path_in'].append(sample_path_in)
                        data_dc['samples_path_out'].append(sample_path_out) 
                        
                    
                data_list.append(data_dc)
            
        logging.info('Complete collecting files of the dataset. Total files:%d' % file_len)

        return data_list
    
    def select_sampling_method(self, mesh):
        '''if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)'''
    
            
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4*self.num_samplemesh_inout)
        sample_points = surface_points + np.random.normal(scale=self.cfg.SAMPLING_SIGMA, size=surface_points.shape)
        

        length = 1
        MIN = -0.5
        
        '''
        #add random points within generating space
        random_points = np.random.rand(self.num_samplemesh_inout // 4, 3)*length + MIN
        sample_points = np.concatenate([sample_points, random_points], axis=0)
        np.random.shuffle(sample_points) #(N,3)
        
        #label the points
        inside_idx = mesh.contains(sample_points)
        inside_points = sample_points[inside_idx]
        outside_points = sample_points[np.logical_not(inside_idx)]'''
        
        #add random points within generating space
        random_points = np.random.rand(self.num_samplemesh_inout // 4, 3)*length + MIN
        sample_points = np.concatenate([sample_points, random_points], axis=0)
        np.random.shuffle(sample_points) #(N,3)
        sub_surface_points = surface_points[:self.num_samplemesh_inout//4,:]
        sample_points = np.concatenate([sub_surface_points, sample_points],axis=0)
        
        #label the points
        inside_idx = mesh.contains(sample_points)
        inside_idx[:self.num_samplemesh_inout//4] = 1
        inside_points = sample_points[inside_idx]
        outside_points = sample_points[np.logical_not(inside_idx)]

        
        n_in = inside_points.shape[0]
        #限制点的个数
        n_s = self.num_samplemesh_inout // 2
        inside_points = inside_points[
            :n_s] if n_in > n_s else inside_points
        outside_points = outside_points[
            :n_s] if n_in > n_s else outside_points[:(self.num_samplemesh_inout-n_in)]
        
        samples = np.concatenate([inside_points, outside_points], 0) #(N,3)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])),
                                 np.zeros((1,outside_points.shape[0]))], axis=1) #(1,N)
        
        #shuffle
        s_idx = np.arange(self.num_samplemesh_inout)
        np.random.shuffle(s_idx)
        samples = samples[s_idx,:]
        labels = labels[:,s_idx]
        #turn into torch tensor
        samples = torch.Tensor(samples).float()
        labels  = torch.Tensor(labels).float()
        
        return {
            'samples': samples,
            'labels': labels
        }
        
    def _get_options(self):
        options = {}
        options['required_items'] = ['partial_cloud', 'sampled_gt_points']
        return options


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        #读取需要获取的文件
        data_dc = self.data_list[index]
        data = {}
        for ri in self.options['required_items']:
            rand_ii = np.random.randint(0, 4, 3)
            if ri == 'sampled_gt_points':#获取采样数据和label
                if  self.cfg. DATASETS.SHAPENETTOUCH.LOAD_MODE == 1:
                    s_idx = np.arange(self.num_samplemesh_inout)
                    np.random.shuffle(s_idx)
                    #print("sampling shape ", data_dc['samples_path_in'].shape, data_dc['samples_path_out'].shape) 

                    samples_path_in = data_dc['samples_path_in'][rand_ii[0]]
                    sample_path_out = data_dc['samples_path_out'][rand_ii[1]]
                    samples_tmp, n_in, n_out = load_gt(samples_path_in, sample_path_out)
                    labels_tmp = np.concatenate([np.ones((1,n_in)), np.zeros((1,n_out))], axis=1)
                    samples = samples_tmp[s_idx, :]
                    labels = labels_tmp[:,s_idx]
                    #print("sampling shape ", samples.shape, labels.shape)
                    #turn into torch tensor
                    samples = torch.Tensor(samples).float()
                    labels  = torch.Tensor(labels).float()
                    sample_data = {
                        'samples_path_in': samples,
                        'samples_path_out' : labels
                    }
                else:
                    #print("sampling surface points: ", data_dc['model_id'])
                    sample_data = self.select_sampling_method(data_dc['gt_mesh'])

                data.update(sample_data)
            elif ri == 'partial_cloud':
                data['partial_cloud'] = data_dc['partial_cloud'][rand_ii[2]]
            else:
                data[ri] = data_dc[ri]
                
        #print(data_dc['taxonomy_id'])
        
        if self.transforms is not None:
            data = self.transforms(data)

        return data_dc['taxonomy_id'], data_dc['model_id'], data


#in this version the dataset is represented as 0-on surface, other-off surface
class ImplicitDataset_onoff(torch.utils.data.dataset.Dataset):
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.subset = phase
        self.is_train = phase == 'train'
        
        self.num_samplemesh_inout = cfg.NETWORK.N_SAMPLING_MESHPOINTS 
        self.options = self._get_options()
        
        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENETTOUCH.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read()) #从ShapeNet.json中读取物体目录（全部物体）

        #获取file_list
        self.data_list = self._get_data(self.cfg, self.subset, n_renderings=self.cfg.DATASETS.SHAPENETTOUCH.N_RENDERINGS)
        self.transforms = self._get_transforms(self.cfg, subset=phase)


    def _get_transforms(self, cfg, subset):
        transforms = []
        if subset == 'train':
            transforms.append({
                'callback':'RandomSamplePoints',
                'parameters':{
                    'n_points':cfg.CONST.N_INPUT_POINTS
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
                    'n_points':cfg.CONST.N_INPUT_POINTS
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
        file_len = 0
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]
            
            
            file_len = file_len + len(samples)
            for s in tqdm(samples, leave=False): #遍历该类的全部文件
                
                gt_mesh_path = cfg.DATASETS.SHAPENETTOUCH.MESH_PATH % (subset, dc['taxonomy_id'], s)
                gt_mesh =  load_trimesh(gt_mesh_path)
                
                for i in range(n_renderings):
                    data_dc = {}
                    data_dc['taxonomy_id'] = dc['taxonomy_id']
                    data_dc['model_id'] = s
                    
                    partial_path = cfg.DATASETS.SHAPENETTOUCH.PARTIAL_POINTS_PATH%(subset, dc['taxonomy_id'], s, i)
                    data_dc['partial_cloud'] = IO.get(partial_path).astype(np.float32)
                    data_dc['gt_mesh'] = gt_mesh
                    
                    data_list.append(data_dc)
            
        logging.info('Complete collecting files of the dataset. Total files:%d' % file_len)

        return data_list
    
    def select_sampling_method(self, mesh):
        '''if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)'''
        #表面采样
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4*self.num_samplemesh_inout)
        sample_points = surface_points + np.random.normal(scale=self.cfg.SAMPLING_SIGMA, size=surface_points.shape)
        
        #add random points within generating space
        length = 1
        MIN = -0.5
        random_points = np.random.rand(self.num_samplemesh_inout // 4, 3)*length + MIN
        sample_points = np.concatenate([sample_points, random_points], axis=0)
        np.random.shuffle(sample_points) #(N,3)
        sample_points = np.concatenate([surface_points, sample_points])
        
        #label the points
        (closest_points, distances, triangle_id) = mesh.nearest.on_surface(sample_points)
        inside_idx = distances < 0.001
        inside_points = sample_points[inside_idx]
        outside_points = sample_points[np.logical_not(inside_idx)]
        
        n_in = inside_points.shape[0]
        #限制点的个数
        n_s = self.num_samplemesh_inout // 2
        inside_points = inside_points[
            :n_s] if n_in > n_s else inside_points
        outside_points = outside_points[
            :n_s] if n_in > n_s else outside_points[:(self.num_samplemesh_inout-n_in)]
        
        samples = np.concatenate([inside_points, outside_points], 0) #(N,3)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])),
                                 np.zeros((1,outside_points.shape[0]))], axis=1) #(1,N)
        
        #shuffle
        s_idx = np.arange(self.num_samplemesh_inout)
        np.random.shuffle(s_idx)
        samples = samples[s_idx,:]
        labels = labels[:,s_idx]
        #turn into torch tensor
        samples = torch.Tensor(samples).float()
        labels  = torch.Tensor(labels).float()
        
        return {
            'samples_path_in': samples,
            'samples_path_out': labels
        }
        
    def _get_options(self):
        options = {}
        options['required_items'] = ['partial_cloud', 'sampled_gt_points']
        return options


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        #读取需要获取的文件
        data_dc = self.data_list[index]
        data = {}
        for ri in self.options['required_items']:
            if ri == 'sampled_gt_points':#获取采样数据和label
                sample_data = self.select_sampling_method(data_dc['gt_mesh'])
                #from mesh_to_sdf import sample_sdf_near_surface
                #sample_data, label = sample_sdf_near_surface(data_dc['gt_mesh'], number_of_points=5000)
                #data['samples_path_in'] = sample_data
                #data['samples_path_out'] = label
                data.update(sample_data)
            else:
                data[ri] = data_dc[ri]
                
        #print(data_dc['taxonomy_id'])
            
        if self.transforms is not None:
            data = self.transforms(data)

        return data_dc['taxonomy_id'], data_dc['model_id'], data