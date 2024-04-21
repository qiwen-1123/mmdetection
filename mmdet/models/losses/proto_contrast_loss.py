from typing import List
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Tuple, Union

import numpy as np
import torch
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss
from mmdet.visualization import show_center, show_conf, show_img


@MODELS.register_module()
class Proto_contrast_loss(nn.Module):
    def __init__(self, human_index:int, loss_weight=1e-2):
        super().__init__()
        self.temp = 7e-2
        self.ce = nn.CrossEntropyLoss()
        self.human_index = human_index
        self.loss_weight = loss_weight
    
    def forward(self, feature:torch.Tensor, center_map: torch.Tensor, score_map: torch.Tensor, cate_protos_pre: torch.Tensor):
        B, E, H, W = feature.shape
        # [B, C, h, w]
        C = score_map.shape[1]
        
        cate_protos_pre = cate_protos_pre.unsqueeze(0).repeat(B, 1, 1)

        score_map = F.interpolate(score_map, (H, W))
        
        # [B, hw, C] 
        score_map = score_map.contiguous().view(B, C, -1).transpose(1, 2).detach()

        # [B, hw, 1] 
        center_map = center_map[:, 0, :, :].contiguous().view(B, -1).unsqueeze(-1).detach().cuda().to(torch.float32)

        # scale up the gap between logits of different classes
        score_map = (score_map / 1e-3).softmax(dim=-1)

        # [B, E, hw]
        feature = feature.contiguous().view(B, E, -1)
        # [B, E, C]
        cate_protos = feature @ score_map
        # Normalize
        cate_protos:torch.Tensor = cate_protos / torch.clamp_min(torch.norm(cate_protos, p=2, dim=1, keepdim=True), 1e-5)

        # Non Ped idxs
        non_ped_idxs = torch.arange(cate_protos.shape[-1])!=self.human_index
        
        # Positive Proto [B, E, 1]
        pos_ped_proto = feature @ center_map
        # Normalize
        pos_ped_proto:torch.Tensor = pos_ped_proto / torch.clamp_min(torch.norm(pos_ped_proto, p=2, dim=1, keepdim=True), 1e-5)

        # Negative Proto [E, B*(C-1)]
        # neg_protos = cate_protos[:, :, non_ped_idxs].transpose(0, 1).contiguous().view(E, B*(C-1))
        neg_protos = cate_protos_pre[:, :, non_ped_idxs].transpose(0, 1).contiguous().view(E, B*(C-1))

        # Normalize [B, E, hw]
        feat_norm:torch.Tensor = feature / torch.clamp_min(torch.norm(feature, p=2, dim=1, keepdim=True), 1e-5)

        # distance between features at all pixels and neg protos in batch dim [B, hw, B*(C-1)]
        feat_neg_score = feat_norm.transpose(1, 2) @ neg_protos

        # select the ped positions [K, 1, B*(C-1)]
        l_neg = feat_neg_score[center_map.squeeze(-1) > 0].unsqueeze(1)

        # distance between features at all pixels and ped proto in batch dim [B, hw, 1]
        feat_ped_score = feat_norm.transpose(1, 2) @ pos_ped_proto

        # select the ped positions [K, 1, 1]
        l_pos = feat_ped_score[center_map.squeeze(-1) > 0].unsqueeze(1)

        # Contrast Logits, from Pos & Neg [K, 1+B*(C-1)]
        contrast_logits = torch.cat([l_pos, l_neg], dim=-1).squeeze(1)

        assert contrast_logits.shape[-1] == 1+B*(C-1)

        K = contrast_logits.shape[0]
        # Label
        labels = torch.zeros(K).long().cuda()

        if contrast_logits.shape[0] <= 0:
            return torch.Tensor([0.0]).cuda()
        else:
            return self.loss_weight*self.ce(contrast_logits/self.temp, labels)

@MODELS.register_module()
class loss_pseudo_score(nn.Module):
    def __init__(self,loss_weight=100):
        super().__init__()
        self.smoothl1 = nn.SmoothL1Loss()
        self.loss_weight = loss_weight

    def forward(self, score_map:torch.Tensor, pesudo_map):
        return self.loss_weight*self.smoothl1(score_map, pesudo_map)