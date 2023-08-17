import trimesh
import os
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


num_samplemesh_inout = 4000
SAMPLING_SIGMA = 0.1
def select_sampling_method(mesh):          
        '''if not is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)'''
        #表面采样
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4*num_samplemesh_inout)
        sample_points = surface_points + np.random.normal(scale=SAMPLING_SIGMA, size=surface_points.shape)
        
        #add random points within generating space
        length = 1
        MIN = -0.5
        random_points = np.random.rand(num_samplemesh_inout // 4, 3)*length + MIN
        sample_points = np.concatenate([sample_points, random_points], axis=0)
        np.random.shuffle(sample_points) #(N,3)
        sub_surface_points = surface_points[:num_samplemesh_inout//4,:]
        sample_points = np.concatenate([sub_surface_points, sample_points],axis=0)
        
        #label the points
        inside_idx = mesh.contains(sample_points)
        inside_idx[:num_samplemesh_inout//4] = 1
        inside_points = sample_points[inside_idx]
        outside_points = sample_points[np.logical_not(inside_idx)]
        
        n_in = inside_points.shape[0]
        #限制点的个数
        n_s = num_samplemesh_inout // 2
        inside_points = inside_points[
            :n_s] if n_in > n_s else inside_points
        outside_points = outside_points[
            :n_s] if n_in > n_s else outside_points[:(num_samplemesh_inout-n_in)]
        
        samples = np.concatenate([inside_points, outside_points], 0) #(N,3)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])),
                                 np.zeros((1,outside_points.shape[0]))], axis=1) #(1,N)
        
        #shuffle
        s_idx = np.arange(num_samplemesh_inout)
        np.random.shuffle(s_idx)
        samples = samples[s_idx,:]
        labels = labels[:,s_idx]

        
        return {
            'samples': samples,
            'labels': labels
        }

def draw_ptcloud(ptcloud, mesh):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = plt.axes(projection='3d')
    ax.axis('off')
    #ax.axis('scaled')
    ax.view_init(30, 45)

    #max, min = np.max(ptcloud), np.min(ptcloud)
    p_max = 0.6
    p_min = -0.6
    ax.set_xbound(p_min, p_max)
    ax.set_ybound(p_min, p_max)
    ax.set_zbound(p_min, p_max)
    
    dx = 1.2
    xx = [p_min, p_max, p_max, p_min, p_min]
    yy = [p_max, p_max, p_min, p_min, p_max]
    kwargs1 = {'linewidth':1, 'color':'black', 'linestyle':'-'}
    kwargs2 = {'linewidth':1, 'color':'black', 'linestyle':'--'}
    ax.plot(xx, yy, p_max, **kwargs1)
    ax.plot(xx[:3], yy[:3], p_min, **kwargs1)
    ax.plot(xx[2:], yy[2:], p_min, **kwargs2)
    for n in range(3):
        ax.plot([xx[n], xx[n]], [yy[n], yy[n]], [p_min, p_max], **kwargs1)
    ax.plot([xx[3], xx[3]], [yy[3], yy[3]], [p_min, p_max], **kwargs2)
    
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
    
    #draw mesh
    xx = mesh.vertices[:,0]
    zz = mesh.vertices[:,1]
    yy = mesh.vertices[:,2]
    tri_idx = mesh.faces
    ax.plot_trisurf(xx, yy, zz, triangles=tri_idx, color=(1,1,1,0.1))

    fig.canvas.draw()
    #img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    #return img
    plt.show()

    

#root_dir = '/home/manager/Desktop/model_manifoldplus.obj'
root_dir = '/home/manager/data/ShapeCompletion/pcd_2048/train/complete/02876657/4301fe10764677dcdf0266d76aa42ba/o1_manifold_plus.obj'
mesh = trimesh.load(root_dir, process=False)

if not mesh.is_watertight:
    face_idx = trimesh.repair.broken_faces(mesh)
    trimesh.repair.fill_holes(mesh)
    print('hole filled: ',mesh.is_watertight)

label_dict = select_sampling_method(mesh)

samples = label_dict['samples']
labels = label_dict['labels']

point_cloud = samples[(labels[0] > 0.5)]

draw_ptcloud(point_cloud, mesh)
