# TouchNet

This repository contains the source code for the paper [End-to-end Surface Reconstruction For Touching Trajectories.].

![Overview](https://infinitescript.com/wordpress/wp-content/uploads/2020/07/TouchNet-Overview.png)

## Cite this work

```
@inproceedings{liu2021touchnet,
  title={End-to-end Surface Reconstruction For Touching Trajectories},
  author={Liu, Jiarui and 
          Zhang, Yuanpei and 
          Zou, Zhuojun and 
          Hao, Jie},
  booktitle={ACCV},
  year={2022}
}
```

## Datasets

Our datasets are available below:

- [Mug]()
- [Airplane]()
- [Car]()

## Pretrained Models

The pretrained models are available as follows:

- [Mug]( ) ( MB)
- [Airplane]( ) ( MB)
- [Car]( ) ( MB)
## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/LiuLiuJerry/ShapeNetProcessing.git 
```
Note that this code is partly based on TouchNet( https://github.com/hzxie/TouchNet )

#### Install Python Denpendencies

```
cd TouchNet
pip install -r requirements.txt
```

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.SHAPENETTOUCH.CATEGORY_FILE_PATH         = 'path/to/datasets/datasetname/ShapeNetTouch.json'
__C.DATASETS.SHAPENETTOUCH.N_RENDERINGS               = 4 #一个物体对应4个rendering
__C.DATASETS.SHAPENETTOUCH.N_POINTS                   = 2048 #好像也没用到？
# --参数1：subset（train/val）, 参数2：dc['taxonomy_id'], 物体类别的id  参数3： 数据id  参数4：i,同一个物体不同的exploration
__C.DATASETS.SHAPENETTOUCH.PARTIAL_POINTS_PATH        = 'path/to/datasets/datasetname/%s/partial/%s/%s/path2048_%.2d.pcd'
__C.DATASETS.SHAPENETTOUCH.COMPLETE_POINTS_PATH       = 'path/to/datasets/datasetname/%s/complete/%s/%s/01.pcd'
__C.DATASETS.SHAPENETTOUCH.MESH_PATH                  = 'path/to/datasets/datasetname/%s/complete/%s/%s/simplified.obj' 
__C.DATASETS.SHAPENETTOUCH.LOAD_MODE                  = 0 # 0:sample online   1:sample offline and load online
__C.DATASETS.SHAPENETTOUCH.N_SAMPLES                  = 20

```

## Get Started

To train TouchNet, you can simply use the following command:

```
python3 runner.py
```

To test TouchNet, you can use the following command:

```
python3 runner.py --test=Ture --weights=/path/to/pretrained/model.pth
```

To generate new mesh from TouchNet, you can use the following command:
```
python3 runner.py --test=False --inference=True --weights=/path/to/pretrained/model.pth
```

