# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import copy

import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmdet3d.models import builder

from projects.mmdet3d_plugin.models.dense_heads.meformer_head import pos2embed


@TRANSFORMER.register_module()
class PMETransformer(BaseModule):
    def __init__(
            self,
            decoder=None,
            heads=None,
            separate_head=None,
            num_classes=None,
            init_cfg=None
    ):
        super(PMETransformer, self).__init__(init_cfg=init_cfg)
        self.dist_scaler = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dist_bias = nn.Parameter(torch.randn(1), requires_grad=True)

        self.decoder = build_transformer_layer_sequence(decoder)

        self.embed_dims = self.decoder.embed_dims
        self.num_layers = decoder["num_layers"]
        self.num_heads = decoder["transformerlayers"]["attn_cfgs"][0]["num_heads"]
        self.box_pos_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.modality_proj = nn.ModuleDict({
            "fused": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            ),
            "bev": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            ),
            "img": nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims)
            )
        })

        self.task_heads = nn.ModuleList()
        for num_cls in num_classes:
            heads = copy.deepcopy(heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=self.embed_dims,
                heads=heads, num_cls=num_cls,
                groups=decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.decoder.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

        self._is_init = True

    def forward(
            self,
            x,
            reference,
            outs,
            modalities,
            num_queries_per_modality,
            task_id,
            pc_range,
            attn_masks=None
    ):
        x = x[-1].transpose(0, 1)
        x = list(x.split(num_queries_per_modality, dim=0))
        x_proj = []

        for i, modality in enumerate(modalities):
            x_proj.append(self.modality_proj[modality](x[i]))

        target = x_proj[modalities.index("fused")]
        memory = torch.cat(x_proj, dim=0)

        center = outs["center"][-1]

        box_pos_embed = pos2embed(center, self.embed_dims)
        box_pos_embed = self.box_pos_embedding(box_pos_embed).transpose(0, 1)
        box_pos_embed = list(box_pos_embed.split(num_queries_per_modality, dim=0))

        query_box_pos_embed = box_pos_embed[modalities.index("fused")]
        key_box_pos_embed = torch.cat(box_pos_embed, dim=0)

        center = list(center.split(num_queries_per_modality, dim=1))
        center_q = center[modalities.index("fused")]
        center_kv = torch.cat(center, dim=1)
        dist = (center_q.unsqueeze(2) - center_kv.unsqueeze(1)).norm(p=2, dim=-1)
        dist_mask = dist * self.dist_scaler + self.dist_bias

        if attn_masks is None:
            attn_masks = torch.zeros((target.shape[0], target.shape[0]), dtype=torch.bool, device=target.device)

        attn_masks = torch.zeros_like(attn_masks, dtype=torch.float).float().masked_fill(attn_masks, float("-inf"))
        attn_masks = attn_masks.repeat(1, len(x_proj))
        attn_masks = attn_masks + dist_mask
        attn_masks = attn_masks.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)

        outs_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_box_pos_embed,
            key_pos=key_box_pos_embed,
            attn_masks=[attn_masks]
        )

        outs_dec = outs_dec.transpose(1, 2)
        outs = self.task_heads[task_id](outs_dec)

        reference = reference.split(num_queries_per_modality, dim=1)[modalities.index("fused")]

        center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
        height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
        _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
        _center[..., 0:1] = center[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        _center[..., 1:2] = center[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        _height[..., 0:1] = height[..., 0:1] * (pc_range[5] - pc_range[2]) + pc_range[2]
        outs['center'] = _center
        outs['height'] = _height

        return outs
