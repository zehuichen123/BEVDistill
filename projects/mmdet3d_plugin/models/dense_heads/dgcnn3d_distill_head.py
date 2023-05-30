import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer
                        
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from .dgcnn3d_head import DGCNN3DHead

from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.builder import build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models import builder

        
@HEADS.register_module()
class DGCNN3DDistillHead(DGCNN3DHead):
    """Head of DeformDETR3D. 
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                num_heatmap_convs=3,
                share_conv_channel=64,
                max_radius=2,
                min_overlap=0.1,
                tasks=None,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                bias='auto',
                loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                **kwargs,):
        super().__init__(**kwargs)
        self.has_heatmap = False
        if tasks is not None:
            self.max_radius = max_radius
            self.has_heatmap = True
            num_classes = [len(t['class_names']) for t in tasks]
            self.class_names = [t['class_names'] for t in tasks]
            
            self.loss_heatmap = build_loss(loss_heatmap)
            self.task_heads = nn.ModuleList()

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
        self.distill_bbox_coder = build_bbox_coder(kwargs['distill_bbox_coder'])

    @force_fp32(apply_to=('preds_dicts'))
    def get_distill_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.distill_bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            if bboxes.size(-1) == 9:
                # NOTE hard code here
                if 3 in img_metas[i]:
                    bboxes = img_metas[i][3]['box_type_3d'](bboxes, 9)
                else:
                    bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            else:
                bboxes = img_metas[i]['box_type_3d'](bboxes, 7)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def forward(self, mlvl_feats):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = self.bev_shape
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, bev_outputs = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_enc=True,
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., 0:2] += reference
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])

            if tmp.size(-1) > 8:
                outputs_coord = torch.cat((tmp[..., :6], tmp[..., 6:8], tmp[..., 8:]), -1)
            else:
                outputs_coord = torch.cat((tmp[..., :6], tmp[..., 6:8]), -1)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.as_two_stage:
            outs = {
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': enc_outputs_class,
                'enc_bbox_preds': enc_outputs_coord.sigmoid(), 
                'bev_outputs': mlvl_feats
            }
        else:
            outs = {
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
                'bev_outputs': mlvl_feats
            }
        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
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
        bev_outputs = preds_dicts['bev_outputs']

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

        losses_cls, losses_bbox = multi_apply(
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
        if self.has_heatmap:
            loss_heatmap = self.get_heatmap_loss(bev_outputs[0], gt_bboxes_list, gt_labels_list, img_metas=img_metas)
            loss_dict.update(loss_heatmap)
        return loss_dict

    def get_heatmap_targets(self, gt_bboxes_3d, gt_labels_3d):
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
                        min_overlap=0.1)
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

    @force_fp32(apply_to=('preds_dicts'))
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
        bev_feat = bev_feat.detach()
        heatmap_list = self.get_heatmap_targets(gt_bboxes_list, gt_labels_list)

        bev_feat, heatmap_pred = self.get_heatmap_pred(bev_feat)
        
        loss_dict = dict()

        num_task = len(heatmap_list)
        for task_id in range(num_task):
            bev_pred = clip_sigmoid(heatmap_pred[task_id]['heatmap'])
            label_weights = bev_feat.new_ones(bev_pred.shape, device=bev_feat.device)
            heatmap = heatmap_list[task_id]
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
        # vis_map = torch.zeros((bs, tmp_h, tmp_w), device=device)
        # for each_heatmap in heatmap_pred:
        #     vis_map = vis_map + each_heatmap['heatmap'].sum(1)

        # vis_map = vis_map.cpu().detach().numpy()
        # import numpy as np
        # import cv2
        # for batch_id in range(bs):
        #     sample_idx = img_metas[batch_id]['sample_idx']
        #     cv2.imwrite('distill_code/data/%s_c.jpg' % sample_idx, vis_map[batch_id].squeeze() * 255)
            
        return loss_dict
            