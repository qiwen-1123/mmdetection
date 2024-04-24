from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.registry import DATASETS

import numpy as np
import torch

@TRANSFORMS.register_module()
class GaussianMapTransform(BaseTransform):
    def __init__(self, dataset_type,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = dataset_type
        self.data_cls = DATASETS.get(self.dataset_type).METAINFO['classes']
        
    def transform(self, results: dict) -> dict:
        """Method to get Gussian Map for construct positive prototype in VLPD.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            results (dict): add Gaussian map
        """
        cls_num = len(self.data_cls)
        pos_map = torch.zeros((cls_num, *results['img_shape']), dtype=torch.float32)
        gt_cls = np.unique(results['gt_bboxes_labels']) # array of gt bbox class index, skip non existing classes
        for idx in gt_cls:
            idx_goal_cls = np.where(results['gt_bboxes_labels'] == idx) 

            gts = results['gt_bboxes'].tensor
            gts = gts[idx_goal_cls]  # the gts of cls

            if len(gts.shape) != 2:
                gts = gts.unsqueeze(0)

            if len(gts) > 0:
                for ind in range(len(gts)):
                    x1, y1, x2, y2 = gts[ind, :4].ceil().int()
                    dx = gaussian(x2 - x1)
                    dy = gaussian(y2 - y1)
                    gau_map = dy @ dx.T
                    pos_map[idx, y1:y2, x1:x2] = torch.maximum(pos_map[idx, y1:y2, x1:x2], gau_map)  # gauss map

        results['gauss'] = pos_map
        return results
    
    def __repr__(self) -> str:
        repr_str = 'gauss loss'
        return repr_str

        
        
def gaussian(kernel):
    sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
    s = 2*(sigma**2)
    dx = torch.exp(-torch.square(torch.arange(kernel).float() - int(kernel / 2)) / s)
    return dx.view(-1, 1)