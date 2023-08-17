from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_ptcloud_img(ptcloud):


    f = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud
    ax = plt.axes(projection='3d')
    #ax.axis('off')
    #ax.axis('scaled')
    ax.view_init(30, 45)

    #p_max, p_min = np.p_max(ptcloud), np.p_min(ptcloud)
    p_max = 0.6
    p_min = -0.6
    '''ax.set_xbound(p_min, p_max)
    ax.set_ybound(p_min, p_max)
    ax.set_zbound(p_min, p_max)'''
    
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
    
    ax.set_xticks(np.linspace(p_min, p_max, 6))
    ax.set_yticks(np.linspace(p_min, p_max, 6))
    ax.set_zticks(np.linspace(p_min, p_max, 6))
    
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    plt.show()

#构建xyz
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)
c = x + y

b = z*0.5

get_ptcloud_img([x,y,z])
get_ptcloud_img([x,y,z*0.5])
