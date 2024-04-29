from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

import numpy as np
import torch

@TRANSFORMS.register_module()
class GaussianMapTransform(BaseTransform):
    def transform(self, results: dict) -> dict:
        """Method to get Gussian Map for construct positive prototype in VLPD.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            results (dict): add Gaussian map
        """

        pos_map = torch.zeros(results['img_shape'], dtype=torch.float32)
        idx_goal_cls = np.where(results['gt_bboxes_labels'] == 0) # 0 is person
        idx_goal_cls = torch.tensor(idx_goal_cls).squeeze()
        
        gts=results['gt_bboxes'].tensor
        gts = gts[idx_goal_cls] # the gts of cls 0
        
        if len(gts.shape) != 2:
            gts = gts.unsqueeze(0)
        
        if len(gts) > 0:
            for ind in range(len(gts)):
                x1, y1, x2, y2 = gts[ind, :4].ceil().int()
                dx = gaussian(x2 - x1)
                dy = gaussian(y2 - y1)
                gau_map = dy @ dx.T
                pos_map[y1:y2, x1:x2] = torch.maximum(pos_map[y1:y2, x1:x2], gau_map)  # gauss map
            
        results['gauss'] = pos_map
        return results
    
    def __repr__(self) -> str:
        repr_str = 'gauss loss'
        return repr_str

        
        
# def gaussian(kernel):
#     sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
#     s = 2*(sigma**2)
#     dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
#     return np.reshape(dx, (-1, 1))

def gaussian(kernel):
    sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
    s = 2*(sigma**2)
    dx = torch.exp(-torch.square(torch.arange(kernel).float() - int(kernel / 2)) / s)
    return dx.view(-1, 1)