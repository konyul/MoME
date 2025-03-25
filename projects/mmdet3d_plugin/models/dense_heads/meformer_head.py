# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import collections
import copy

import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean, build_bbox_coder
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models import builder

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


@HEADS.register_module()
class MEFormerHead(BaseModule):
    def __init__(self,
                 in_channels,
                 use_ensemble=False,
                 ensemble=None,
                 num_query=900,
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 pc_range=[],
                 modalities=dict(),
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                     type="L1Loss",
                     reduction="mean",
                     loss_weight=0.25,
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        super(MEFormerHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_ensemble = use_ensemble
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.pc_range = pc_range
        self.modalities = modalities
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.fp16_enabled = False

        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )

        # transformer
        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

        if self.use_ensemble:
            ensemble["heads"] = common_heads
            ensemble["separate_head"] = separate_head
            ensemble["num_classes"] = self.num_classes
            self.ensemble = build_transformer(ensemble)
        else:
            self.ensemble = None

        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def init_weights(self):
        super(MEFormerHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0)  # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :]) \
                    / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None,
                      None, :]
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(
            self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        proj_points = torch.einsum('bnd, bvcd -> bvnc',
                                   torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1),
                                   lidars2imgs)

        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (
                proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape)

        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (
                proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (
                self.pc_range[3] - 1) / self.depth_num
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        proj_points_clone = torch.cat(
            [proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None,
                                                      :]) \
                          / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(
            self.pc_range[:3]))[None, None, None, :]

        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        rv_embeds = self._rv_query_embed(ref_points, img_metas)
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        ret_dicts = []
        x = self.shared_conv(x)

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)

        modalities = copy.deepcopy(self.modalities["train" if self.training else "test"])
        outs_dec = self.transformer(
            x, x_img, bev_query_embeds, rv_query_embeds, bev_pos_embeds, rv_pos_embeds, img_metas,
            attn_masks=attn_mask, modalities=modalities
        )
        num_queries_per_modality = [m.shape[2] for m in outs_dec]
        outs_dec = torch.cat(outs_dec, dim=2)
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        reference = reference.repeat(1, len(modalities), 1)

        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)

            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height

            if self.use_ensemble and "fused" in modalities:
                pc_range = torch.Tensor(self.pc_range).to(reference.device)
                outs = self.ensemble(
                    outs_dec,
                    reference,
                    outs,
                    modalities,
                    num_queries_per_modality,
                    task_id,
                    pc_range,
                    attn_mask
                )

                modalities = ["ensemble"]
                num_queries_per_modality = [outs["center"].shape[2]]

            if self.training:
                for key in outs.keys():
                    outs[key] = list(outs[key].split(num_queries_per_modality, dim=2))

            outs["modalities"] = modalities

            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    if key not in ["modalities"]:
                        outs["dn_" + key] = []
                        for i in range(len(outs[key])):
                            outs['dn_' + key].append(outs[key][i][:, :, :mask_dict['pad_size'], :])
                            outs[key][i] = outs[key][i][:, :, mask_dict['pad_size']:, :]

                outs['dn_mask_dict'] = task_mask_dict

            ret_dicts.append(outs)

        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas)

    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:

            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes,),
                                        num_classes,
                                        dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0

            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks \
            = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)

        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append(
                [labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append(
                [bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append(
                [bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])

        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)

    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task,
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']

        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()

        if task_mask_sum > 0:
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split

        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        loss_bbox = self.loss_bbox(
            pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_tgt)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = dict()
        for i, modality in enumerate(preds_dicts[0][0]["modalities"]):
            num_decoder = preds_dicts[0][0]['center'][i].shape[0]
            all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

            for task_id, preds_dict in enumerate(preds_dicts, 0):
                for dec_id in range(num_decoder):
                    pred_bbox = [preds_dict[0]['center'][i][dec_id], preds_dict[0]['height'][i][dec_id],
                                 preds_dict[0]['dim'][i][dec_id], preds_dict[0]['rot'][i][dec_id]]
                    if 'vel' in preds_dict[0]:
                        pred_bbox.append(preds_dict[0]['vel'][i][dec_id])
                    pred_bbox = torch.cat(pred_bbox, dim=-1)
                    all_pred_bboxes[dec_id].append(pred_bbox)
                    all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][i][dec_id])
            all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
            all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

            loss_cls, loss_bbox = multi_apply(
                self.loss_single, all_pred_bboxes, all_pred_logits,
                [gt_bboxes_3d for _ in range(num_decoder)],
                [gt_labels_3d for _ in range(num_decoder)],
            )

            loss_dict[f'loss_cls_{modality}'] = loss_cls[-1]
            loss_dict[f'loss_bbox_{modality}'] = loss_bbox[-1]

            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                               loss_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls_{modality}'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.loss_bbox_{modality}'] = loss_bbox_i
                num_dec_layer += 1

            dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
            dn_mask_dicts = collections.defaultdict(list)
            for task_id, preds_dict in enumerate(preds_dicts, 0):
                for dec_id in range(num_decoder):
                    pred_bbox = [preds_dict[0]['dn_center'][i][dec_id], preds_dict[0]['dn_height'][i][dec_id],
                                 preds_dict[0]['dn_dim'][i][dec_id], preds_dict[0]['dn_rot'][i][dec_id]]
                    if 'vel' in preds_dict[0]:
                        pred_bbox.append(preds_dict[0]['dn_vel'][i][dec_id])
                    pred_bbox = torch.cat(pred_bbox, dim=-1)
                    dn_pred_bboxes[dec_id].append(pred_bbox)
                    dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][i][dec_id])
                    dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
            dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
            dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
            dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
            dn_loss_cls, dn_loss_bbox = multi_apply(
                self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
            )

            loss_dict[f'dn_loss_cls_{modality}'] = dn_loss_cls[-1]
            loss_dict[f'dn_loss_bbox_{modality}'] = dn_loss_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                               dn_loss_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls_{modality}'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox_{modality}'] = loss_bbox_i
                num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
