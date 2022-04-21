import logging 
import open3d

from extensions.chamfer_dist import ChamferDistance
import torch.nn as nn

class Metrics(object):
    #静态成员变量
    ITEMS = [{
        'name': 'MSE',
        'enabled': True,
        'eval_func':'cls._get_mse',
        'eval_object': nn.MSELoss(),
        'is_greater_better': True,
        'init_value' : 5000
    },{
        'name': 'ChamferDistance',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distance',
        'eval_object': ChamferDistance(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'ChamferDistance_cloud',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distance_cloud',
        'eval_object': ChamferDistance(ignore_zeros=True),
        'is_greater_better': True,
        'init_value': 0
    }, 
    {
        'name': 'F-Score_cloud_0.01',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'F-Score_cloud_0.02',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }]
    

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]
    
    @classmethod
    def get(cls, pred, gt, sample_cloud):
        pt_cloud = sample_cloud[pred[:,-1,:] > 0.5].unsqueeze(0)#暂时假设BN=1
        gt_cloud = sample_cloud[gt[:,-1,:] > 0.5].unsqueeze(0)
        
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            if item['name'].find('cloud') > 0:
                eval_func = eval(item['eval_func'])
                _values[i] = eval_func(pt_cloud, gt_cloud)
            else:
                eval_func = eval(item['eval_func'])
                _values[i] = eval_func(pred, gt)          
       
        return _values
    

    @classmethod
    def _get_mse(cls, pred, gt):
        return cls.ITEMS[0]['eval_object'](gt, pred).item() * 1000
    
    @classmethod
    def _get_chamfer_distance(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(
            pred.transpose(2,1), gt.transpose(2,1)).item() * 1000
    
    @classmethod
    def _get_chamfer_distance_cloud(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000
    
    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        pred = cls._get_open3d_ptcloud(pred)
        gt = cls._get_open3d_ptcloud(gt)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0
    

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud


    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
