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
        self.queue_size = 1024
        #register queue to save all cate_protos
        self.register_buffer("queue",  nn.functional.normalize(torch.randn(self.queue_size, 256, 80, requires_grad=False), dim=1)/10)
        self.ptr = 0
        self.queue_is_full = False

    @torch.no_grad()
    def update_queue(self, k):
        """ swap oldest batch with the current key batch and update ptr"""
        batch_size = k.shape[0]
        if (self.ptr + batch_size)>=self.queue_size and self.queue_is_full==False:
            self.queue_is_full = True
        self.queue[self.ptr: self.ptr + batch_size, :] = k.detach().cpu()
        self.ptr = (self.ptr + batch_size) % self.queue_size
        self.queue.requires_grad = False
        
    
    def forward(self, feature:torch.Tensor, center_map: torch.Tensor, score_map: torch.Tensor):
        B, E, H, W = feature.shape
        # [B, C, h, w]
        C = score_map.shape[1]  # num of classes in clip scoremap (dataset Class + extra background)
        Class = center_map.shape[1] # num of classes in dataset

        score_map = F.interpolate(score_map, (H, W))
        
        # [B, hw, C] 
        score_map = score_map.contiguous().view(B, C, -1).transpose(1, 2).detach()

        # [B, hw, Class] 
        center_map = center_map[:, :, :, :].contiguous().view(B, Class, -1).transpose(1, 2).detach().cuda().to(torch.float32)

        # scale up the gap between logits of different classes
        score_map = (score_map / 1e-3).softmax(dim=-1)

        # [B, E, hw]
        feature = feature.contiguous().view(B, E, -1)
        # [B, E, C]
        cate_protos = feature @ score_map
        # Normalize
        cate_protos:torch.Tensor = cate_protos / torch.clamp_min(torch.norm(cate_protos, p=2, dim=1, keepdim=True), 1e-5)

        # update queue
        self.update_queue(cate_protos)

        # Positive Proto [B, E, Class]
        pos_proto = feature @ center_map
        # Normalize
        pos_proto:torch.Tensor = pos_proto / torch.clamp_min(torch.norm(pos_proto, p=2, dim=1, keepdim=True), 1e-5)

        # Negative Proto [E, B*(C-1)]
        if self.queue_is_full!=True:
            Q = self.ptr
        else:
            Q = self.queue_size
            
        neg_protos = self.queue[:Q, :, :].transpose(0, 1).contiguous().view(E, Q*(C))

        # Normalize [B, E, hw]
        feat_norm:torch.Tensor = feature / torch.clamp_min(torch.norm(feature, p=2, dim=1, keepdim=True), 1e-5)

        # distance between features at all pixels and neg protos in batch dim [B, hw, Q*C]
        feat_neg_score = feat_norm.transpose(1, 2) @ neg_protos
        # distance between features at all pixels and pos proto in batch dim [B, hw, Class]
        feat_pos_score = feat_norm.transpose(1, 2) @ pos_proto
        
        proto_loss = torch.tensor(0, dtype=torch.float32).cuda()
        for cls in range(Class):
            # select the pos positions [K, 1, B*(C-1)], K is num of pixels inside the bbox
            center_map_cls=center_map[:,:,cls]
            if (center_map_cls > 0).sum() <=0: # no gt bbox for this cls, skip
                continue
            # remove the pos cls in the feat_neg_score
            feat_neg_score_cls = feat_neg_score.clone()
            mask_neg = torch.ones_like(feat_neg_score_cls, dtype=torch.bool)
            pos_idx = torch.arange(Q)*Class+cls
            mask_neg[:,:,  pos_idx] = False
            # select the neg reagrding to cls as pos, in batch dim [B, hw, Q*(C-1)]
            feat_neg_score_cls = torch.masked_select(feat_neg_score_cls, mask_neg).view(B, H*W, -1)
            
            # select the pos positions [K, 1, Q*(C-1)]
            l_neg = feat_neg_score_cls[center_map_cls > 0].unsqueeze(1)

            # select the pos positions [K, 1, 1]
            l_pos = feat_pos_score[center_map_cls> 0][:,cls].unsqueeze(1).unsqueeze(1)

            # Contrast Logits, from Pos & Neg [K, 1+Q*(C-1)]
            contrast_logits = torch.cat([l_pos, l_neg], dim=-1).squeeze(1)

            assert contrast_logits.shape[-1] == 1+Q*(C-1)

            K = contrast_logits.shape[0]
            # Label
            labels = cls*torch.ones(K).long().cuda()
            
            if contrast_logits.shape[0] <= 0:
                continue
            else:
                proto_loss+=self.ce(contrast_logits/self.temp, labels)

        return self.loss_weight*proto_loss

@MODELS.register_module()
class loss_pseudo_score(nn.Module):
    def __init__(self,loss_weight=100):
        super().__init__()
        self.smoothl1 = nn.SmoothL1Loss()
        self.loss_weight = loss_weight

    def forward(self, score_map:torch.Tensor, pesudo_map):
        return self.loss_weight*self.smoothl1(score_map, pesudo_map)