# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER


@TRANSFORMER.register_module()
class MOADTransformer(BaseModule):
    def __init__(
            self,
            use_type_embed=True,
            use_cam_embed=False,
            encoder=None,
            decoder=None,
            init_cfg=None,
            cross=False
    ):
        super(MOADTransformer, self).__init__(init_cfg=init_cfg)

        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.use_type_embed = use_type_embed
        self.use_cam_embed = use_cam_embed

        if self.use_type_embed:
            self.bev_type_embed = nn.Parameter(torch.randn(self.embed_dims))
            self.rv_type_embed = nn.Parameter(torch.randn(self.embed_dims))
        else:
            self.bev_type_embed = None
            self.rv_type_embed = None

        if self.use_cam_embed:
            self.cam_embed = nn.Sequential(
                nn.Conv1d(16, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims),
                nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.BatchNorm1d(self.embed_dims)
            )
        else:
            self.cam_embed = None

        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, x_img, bev_query_embed, rv_query_embed, bev_pos_embed, rv_pos_embed, img_metas,
                attn_masks=None, modalities=None, reg_branch=None):
        bs, c, h, w = x.shape
        bev_memory = rearrange(x, "bs c h w -> (h w) bs c")  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)

        bev_pos_embed = bev_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)

        if self.use_type_embed:
            bev_query_embed = bev_query_embed + self.bev_type_embed
            rv_query_embed = rv_query_embed + self.rv_type_embed

        if self.use_cam_embed:
            imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
            imgs2lidars = torch.from_numpy(imgs2lidars).float().to(x.device)
            imgs2lidars = imgs2lidars.flatten(-2).permute(0, 2, 1)
            imgs2lidars = self.cam_embed(imgs2lidars)
            imgs2lidars = imgs2lidars.permute(0, 2, 1).reshape(-1, self.embed_dims, 1, 1)
            imgs2lidars = imgs2lidars.repeat(1, 1, *x_img.shape[-2:])
            imgs2lidars = rearrange(imgs2lidars, '(bs v) c h w -> (v h w) bs c', bs=bs)

        out_decs = []
        for modality in modalities:
            if modality == "fused":
                memory, pos_embed = (torch.cat([bev_memory, rv_memory], dim=0),
                                     torch.cat([bev_pos_embed, rv_pos_embed], dim=0))
                memory_v = memory
                query_embed = bev_query_embed + rv_query_embed
            elif modality == "bev":
                memory, pos_embed = bev_memory, bev_pos_embed
                memory_v = memory
                query_embed = bev_query_embed
            else:
                memory, pos_embed = rv_memory, rv_pos_embed
                memory_v = memory
                if self.cam_embed is not None:
                    memory_v = memory_v * imgs2lidars
                query_embed = rv_query_embed

            query_embed = query_embed.transpose(0, 1)  # [bs, num_query, dim] -> [num_query, bs, dim]
            target = torch.zeros_like(query_embed)

            # out_dec: [num_layers, num_query, bs, dim]
            out_dec = self.decoder(
                query=target,
                key=memory,
                value=memory_v,
                query_pos=query_embed,
                key_pos=pos_embed,
                attn_masks=[attn_masks, None],
                reg_branch=reg_branch,
            )
            out_decs.append(out_dec.transpose(1, 2))

        return out_decs
