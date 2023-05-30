# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor

from .bevformer_head import BEVFormerHead

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet3d.models.builder import build_loss
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from os import path as osp
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet3d.models import builder
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.utils import clip_sigmoid

@HEADS.register_module()
class BEVFormerDistillHead(BEVFormerHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                num_heatmap_convs=3,
                share_conv_channel=64,
                tasks=None,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                bias='auto',
                max_radius=1,
                min_overlap=0.1,
                loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                **kwargs,):
        super().__init__(**kwargs)
        if tasks is not None:
            num_classes = [len(t['class_names']) for t in tasks]
            self.class_names = [t['class_names'] for t in tasks]
            
            self.loss_heatmap = build_loss(loss_heatmap)
            self.task_heads = nn.ModuleList()
            self.max_radius = max_radius
            self.min_overlap = min_overlap

            in_channels = kwargs['in_channels']

            self.shared_conv = ConvModule(
                in_channels,
                share_conv_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=bias)
            separate_head = dict(type='SeparateHead', init_bias=2.19, final_kernel=3)
            for num_cls in num_classes:
                heads = dict(heatmap=(num_cls, num_heatmap_convs))
                separate_head.update(
                    in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
                self.task_heads.append(builder.build_head(separate_head))

    def _build_distill_modules(self, **kwargs):
        distill_assigner = kwargs['distill_assigner']
        self.distill_assigner = build_assigner(distill_assigner)
        distill_sampler_cfg = dict(type='PseudoSampler')
        self.distill_sampler = build_sampler(distill_sampler_cfg, context=self)
        self.loss_cls_distill = build_loss(kwargs['loss_cls_distill'])
        self.loss_bbox_distill = build_loss(kwargs['loss_bbox_distill'])
        self.loss_iou_distill = build_loss(kwargs['loss_iou_distill'])
        self.score_reweight = kwargs.get('score_reweight', False)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        sampling_points = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            feat_point = tmp[..., [0, 1]].detach()
            feat_point = (feat_point - 0.5) * 2

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            sampling_points.append(feat_point)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        sampling_points = torch.stack(sampling_points)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'sampling_points': sampling_points
        }

        return outs

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        reweight_bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], reweight_bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, bbox_weights

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None,
             return_weight=False):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, bbox_weights_list = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        if return_weight:
            return loss_dict, bbox_weights_list
        return loss_dict

    def get_heatmap_targets(self, gt_bboxes_3d, gt_labels_3d, device):
        gt_bboxes_3d = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_3d]
        heatmaps = multi_apply(
            self.get_heatmap_targets_single, gt_bboxes_3d, gt_labels_3d)
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        return heatmaps


    def get_heatmap_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        device = gt_labels_3d.device
        # gt_bboxes_3d = torch.cat(
        #     (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
        #     dim=1).to(device)
        # max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
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
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps = []

        for idx, class_name in enumerate(self.class_names):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))
            num_objs = task_boxes[idx].shape[0]
            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.min_overlap)
                    radius = max(self.max_radius, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

            heatmaps.append(heatmap)
        return heatmaps

    def get_heatmap_pred(self, bev_feat):
        shared_bev_feat = self.shared_conv(bev_feat)
        heatmap_pred = []
        for task in self.task_heads:
            heatmap_pred.append(task(shared_bev_feat))
        return shared_bev_feat, heatmap_pred

    @force_fp32(apply_to=('bev_feat'))
    def get_heatmap_loss(self, bev_feat, gt_bboxes_list, gt_labels_list, img_metas=None):
        # level_index = [128, 64, 32, 16]
        # bev_level_feats = []; start_idx = 0
        # _, bs, in_channels = bev_feat.shape
        # for ii in range(len(level_index)):
        #     bev_level_feats.append(bev_feat[start_idx:start_idx + level_index[ii] ** 2]\
        #                         .permute(0, 2, 1).reshape(bs, -1, level_index[ii], level_index[ii]))
        #     start_idx += level_index[ii] ** 2
        # bev_feat = bev_level_feats[0]
        # simply use the top level 128 x 128
        device = bev_feat.device
        
        heatmap_list = self.get_heatmap_targets(gt_bboxes_list, gt_labels_list, device)

        bev_feat, heatmap_pred = self.get_heatmap_pred(bev_feat)
        
        loss_dict = dict()

        num_task = len(heatmap_list)
        for task_id in range(num_task):
            bev_pred = clip_sigmoid(heatmap_pred[task_id]['heatmap'])
            label_weights = bev_feat.new_ones(bev_pred.shape, device=bev_feat.device)
            heatmap = heatmap_list[task_id]
            # # resize heatmap if the shapes are not equal
            # _, _, s_w, s_h = bev_feat.shape
            # if s_w != heatmap.shape[-1]:
            #     heatmap = F.interpolate(heatmap, (s_w, s_h), mode='bilinear')

            num_pos = heatmap.eq(1).float().sum().item()
            heatmap_loss = self.loss_heatmap(
                bev_pred,
                heatmap, 
                label_weights,
                avg_factor=max(num_pos, 1)
            )
            loss_dict['task%d.loss_heatmap' % task_id] = heatmap_loss

        # device = bev_feat.device
        # bs, _, tmp_h, tmp_w = heatmap_pred[0]['heatmap'].shape
        # # visualize prediction
        # vis_map = torch.zeros((bs, tmp_h, tmp_w), device=device)
        # for each_heatmap in heatmap_pred:
        #     vis_map = vis_map + each_heatmap['heatmap'].sum(1)

        # vis_map = vis_map.cpu().detach().numpy()
        # import numpy as np
        # import cv2
        # for batch_id in range(bs):
        #     sample_idx = img_metas[batch_id]['sample_idx']
        #     cv2.imwrite('distill_code/data/%s_c.jpg' % sample_idx, vis_map[batch_id].squeeze() * 255)

        # vis_map = torch.zeros((bs, tmp_h, tmp_w), device=device)
        # for each_heatmap in heatmap_list:
        #     vis_map = vis_map + each_heatmap.sum(1)

        # vis_map = vis_map.cpu().detach().numpy()
        # import numpy as np
        # import cv2
        # for batch_id in range(bs):
        #     sample_idx = img_metas[batch_id]['sample_idx']
        #     cv2.imwrite('distill_code/data/%s.jpg' % sample_idx, vis_map[batch_id].squeeze() * 255)

        return loss_dict, bev_feat, heatmap_pred

    @force_fp32(apply_to=('preds_dicts'))
    def distill_losses(self,
                    gt_bboxes_list,
                    gt_labels_list,
                    preds_dicts,
                    teacher_bbox_weights_list=None,
                    gt_bboxes_ignore=None,
                    img_metas=None):
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_img_meta_list = [img_metas[0] for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        if teacher_bbox_weights_list is None:
            teacher_bbox_weights_list = [None for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.distill_loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            teacher_bbox_weights_list,
            all_gt_bboxes_ignore_list,
            all_img_meta_list
            )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls_distill'] = enc_loss_cls
            loss_dict['enc_loss_bbox_distill'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls_distill'] = losses_cls[-1]
        loss_dict['loss_bbox_distill'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_distill'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_distill'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def distill_loss_single(self,
                            cls_scores,
                            bbox_preds,
                            gt_bboxes_list,
                            gt_labels_list,
                            teacher_bbox_weights=None,
                            gt_bboxes_ignore_list=None,
                            img_meta=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_distill_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list, img_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0).unsqueeze(-1)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        
        if teacher_bbox_weights is not None:
            bbox_weights = teacher_bbox_weights * bbox_weights
            label_weights = torch.sum(bbox_weights, dim=-1, keepdims=True) != 0
            cls_avg_factor = torch.sum(label_weights)
            num_total_pos = torch.sum(label_weights)

        # cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls_distill(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        if self.score_reweight:
            reweight_score = torch.max(labels, dim=-1, keepdims=True)[0]
            bbox_weights = bbox_weights * reweight_score
            teacher_fg_idx = labels[:, 0] != 10
            num_total_pos = torch.sum(reweight_score[teacher_fg_idx])

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        if self.score_reweight:
            bbox_weights = bbox_weights * reweight_score

        loss_bbox = self.loss_bbox_distill(
                bbox_preds[isnotnan, :self.code_size], normalized_bbox_targets[isnotnan, :self.code_size], bbox_weights[isnotnan, :self.code_size], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def get_distill_targets(self,
                            cls_scores_list,
                            bbox_preds_list,
                            gt_bboxes_list,
                            gt_labels_list,
                            gt_bboxes_ignore_list=None,
                            img_metas=None):
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        img_meta_list = [
            img_metas for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_distill_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, img_meta_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    
    @force_fp32()
    def _get_distill_target_single(self,
                                   cls_score,
                                   bbox_pred,
                                   gt_labels,
                                   gt_bboxes,
                                   gt_bboxes_ignore=None,
                                   img_meta=None):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.distill_assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.distill_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, self.num_classes),
                                    self.num_classes,
                                    dtype=torch.float)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size - 1]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # # NOTE NOTE visualize the matching results
        # pos_cls_score = cls_score[pos_inds]
        # pos_bbox_pred = bbox_pred[pos_inds]
        # gt_bbox_pred = sampling_result.pos_gt_bboxes.cpu().detach().numpy()

        # bboxes = denormalize_bbox(pos_bbox_pred, self.pc_range)[:, :self.code_size - 1].cpu().detach().numpy()
        # # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        # # bboxes = LiDARInstance3DBoxes(bboxes, self.code_size - 1)

        # # gt_bbox_pred[:, 2] = gt_bbox_pred[:, 2] - gt_bbox_pred[:, 5] * 0.5
        # # gt_bboxes = LiDARInstance3DBoxes(gt_bbox_pred, self.code_size - 1)
        
        # from mmdet3d.core.visualizer.show_result import _write_oriented_bbox
        # import os

        # filename = img_meta['sample_idx']
        # result_path = 'distill_code/data/' + filename
        # os.makedirs(result_path, exist_ok=True)
        # # bottom center to gravity center
        # # gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
        # gt_bbox_pred[:, 6] *= -1
        # _write_oriented_bbox(gt_bbox_pred,
        #                      osp.join(result_path, f'{filename}_gt.obj'))
        
        # # bboxes[..., 2] += bboxes[..., 5] / 2
        # bboxes[:, 6] *= -1
        # _write_oriented_bbox(bboxes,
        #                      osp.join(result_path, f'{filename}_pred.obj'))

        # NOTE
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)