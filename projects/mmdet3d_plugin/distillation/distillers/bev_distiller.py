import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict, _load_checkpoint_with_prefix
from mmcv.runner import force_fp32, auto_fp16
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.builder import build_loss
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.utils import clip_sigmoid
from projects.mmdet3d_plugin.models.builder import build_cl_head

@DISTILLER.register_module()
class BEVDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 use_student_assign=False,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 student_pretrained=None,
                 distill_bbox_coder=None,
                 loss_iou_distill=None,
                 loss_cls_distill=None,
                 loss_bbox_distill=None,
                 loss_bev_distill=None,
                 center_aux_loss=False,
                 use_heatmap_feat=False,
                 cl_head=None,
                 loss_heatmap_distill=None,
                 distill_assigner=None,
                 init_student=False,
                 score_reweight=False,
                 distill_bbox=False,
                 grid_size=(1024, 1024),
                 voxel_size=(0.1, 0.1),
                 add_lateral=False,
                 align_type='student'):

        super(BEVDistiller, self).__init__()
        assert align_type in ['student', 'teacher']
        self.align_type = align_type
        self.add_lateral = add_lateral
        if self.add_lateral:
            self.align_conv = nn.Conv2d(
                256,
                256,
                1,
                padding=0)
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.fp16_enabled = False

        self.use_heatmap_feat = use_heatmap_feat
        self.center_aux_loss = center_aux_loss
        if use_heatmap_feat == True and center_aux_loss == False:
            print("DO NOT Support distillation without aux loss on heatmap feature")
            exit()
        self.distill_bbox = distill_bbox
        self.use_student_assign = use_student_assign
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.teacher.pts_bbox_head._build_distill_modules(
            distill_bbox_coder=distill_bbox_coder,
        )
        self.init_weights_teacher(teacher_pretrained)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        self.student.pts_bbox_head._build_distill_modules(
            distill_assigner=distill_assigner,
            loss_iou_distill=loss_iou_distill,
            loss_cls_distill=loss_cls_distill,
            loss_bbox_distill=loss_bbox_distill,
            score_reweight=score_reweight,
        )
        self.loss_heatmap_distill = loss_heatmap_distill
        self.loss_bev_distill = loss_bev_distill
        self.cl_head = build_cl_head(cl_head) if cl_head is not None else cl_head

        self.init_weights_student(student_pretrained)
        if init_student:
            t_checkpoint = _load_checkpoint_with_prefix(prefix='pts_bbox_head.transformer.decoder',filename=teacher_pretrained, map_location='cpu')

            load_state_dict(self.student, t_checkpoint)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward

    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def init_weights_student(self, path=None):
        checkpoint = load_checkpoint(self.student, path, map_location='cpu')

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        student_bev_feat = None
        teacher_bev_feat = None
        with torch.no_grad():
            self.teacher.eval()
            _, teacher_pts_feats = self.teacher.extract_feat(kwargs['points'], img=None, img_metas=None)
            teacher_out = self.teacher.pts_bbox_head(teacher_pts_feats)
            teacher_bbox_list = self.teacher.pts_bbox_head.get_distill_bboxes(teacher_out, img_metas=img_metas)
            teacher_bev_feat_pre = teacher_out['bev_outputs'][0]
            if self.center_aux_loss:
                teacher_bev_feat, teacher_heatmap_pred = self.teacher.pts_bbox_head.get_heatmap_pred(teacher_bev_feat_pre)
            assert len(teacher_bbox_list) == 1
            teacher_preds = teacher_bbox_list[0] # (bboxes, scores, labels)
            teacher_scores = [teacher_preds[1]]
            teacher_bboxes = [teacher_preds[0]]

        student_out = self.student.forward_train(
            img=img, 
            img_metas=img_metas, 
            **kwargs, 
            return_pred=True, 
            return_weight=self.use_student_assign)
        if self.use_student_assign:
            student_loss, student_out, student_weights = student_out
        else:
            student_loss, student_out = student_out
            student_weights = None
        
        # vanilla L2 loss between bevmap
        student_bev_feat_pre = self.convert_student_bev_feat(student_out['bev_embed'])
        student_sampling_points = student_out['sampling_points']
        if self.center_aux_loss:
            student_heatmap_loss, student_bev_feat, student_heatmap_pred = self.student.pts_bbox_head.\
                get_heatmap_loss(
                student_bev_feat_pre, kwargs['gt_bboxes_3d'], kwargs['gt_labels_3d'], img_metas
            )
            student_loss.update(student_heatmap_loss)
        if student_bev_feat is None:
            student_bev_feat = student_bev_feat_pre
        if teacher_bev_feat is None:
            teacher_bev_feat = teacher_bev_feat_pre
        student_bev_feat, teacher_bev_feat = self.unify_feat_size(student_bev_feat, teacher_bev_feat)

        if self.loss_bev_distill is not None:
            if self.loss_bev_distill['fg_weight'] == 'gt':
                bev_fit_loss, fg_map = self.get_bev_fit_loss(student_bev_feat, teacher_bev_feat, gt_bboxes_list=kwargs['gt_bboxes_3d'])
            elif self.loss_bev_distill['fg_weight'] == 'pred':
                bev_fit_loss, fg_map = self.get_bev_fit_loss(student_bev_feat, teacher_bev_feat, fg_pred_map=student_heatmap_pred[0]['heatmap'])
            elif self.loss_bev_distill['fg_weight'] == 'point':
                bev_fit_loss, _ = self.get_bev_fit_loss(student_bev_feat, teacher_bev_feat, sampling_points=student_sampling_points)
            else:
                bev_fit_loss, fg_map = self.get_bev_fit_loss(student_bev_feat, teacher_bev_feat)
            bev_fit_loss['bev_fit_loss'] = bev_fit_loss['bev_fit_loss'] * self.loss_bev_distill['loss_weight']
            student_loss.update(bev_fit_loss)
        if self.loss_heatmap_distill is not None:
            heatmap_fit_loss = self.get_heatmap_distill_loss(student_heatmap_pred, teacher_heatmap_pred, fg_map=fg_map)
            heatmap_fit_loss['heatmap_kl_loss'] = heatmap_fit_loss['heatmap_kl_loss'] * self.loss_heatmap_distill['loss_weight']
            student_loss.update(heatmap_fit_loss)
        if self.distill_bbox:
            pred_distill_loss = self.student.pts_bbox_head.distill_losses(teacher_bboxes, teacher_scores, student_out, student_weights, None, img_metas)
            student_loss.update(pred_distill_loss)

        if self.cl_head is not None:
            if self.use_heatmap_feat == False:
                bev_feat_s = student_bev_feat_pre
                bev_feat_t = teacher_bev_feat_pre
            else:
                bev_feat_s = student_bev_feat
                bev_feat_t = teacher_bev_feat
            relation_distill_loss = self.get_relation_distill_loss(bev_feat_s, bev_feat_t, sampling_points=student_sampling_points)
            student_loss.update(relation_distill_loss)

        return student_loss

    def unify_feat_size(self, student_feat, teacher_feat):
        bs, s_c, s_w, s_h = student_feat.shape
        bs, t_c, t_w, t_h = teacher_feat.shape
        if s_w == t_w:
            return student_feat, teacher_feat
        else:
            teacher_feat = F.interpolate(teacher_feat, (s_w, s_h), mode='bilinear')
            return student_feat, teacher_feat

    @force_fp32(apply_to=('student_bev_feat', 'teacher_bev_feat', 'sampling_points'))
    def get_relation_distill_loss(self, student_bev_feat, teacher_bev_feat, sampling_points=None):
        if sampling_points is not None:
            sampling_points = sampling_points.permute(1, 0, 2, 3)
            select_student_feat = F.grid_sample(student_bev_feat, sampling_points)
            select_teacher_feat = F.grid_sample(teacher_bev_feat, sampling_points)
            # TODO: if use the last stage query points or all 6 stages query points
            # NOTE NOTE
            bs, n_channels, num_stages, num_queries = select_student_feat.shape
            select_student_feat = select_student_feat.permute(0, 2, 3, 1).reshape(bs * num_stages * num_queries, n_channels)
            select_teacher_feat = select_teacher_feat.permute(0, 2, 3, 1).reshape(bs * num_stages * num_queries, n_channels)

            relation_distill_loss = self.cl_head(select_student_feat, select_teacher_feat)
            return dict(relation_distill_loss = relation_distill_loss)

    @force_fp32(apply_to=('student_bev_feat', 'teacher_bev_feat', 'sampling_points'))
    def get_bev_fit_loss(self, student_bev_feat, teacher_bev_feat, gt_bboxes_list=None, fg_pred_map=None, sampling_points=None):
        if sampling_points is not None:
            sampling_points = sampling_points.permute(1, 0, 2, 3)
            select_student_feat = F.grid_sample(student_bev_feat, sampling_points)
            select_teacher_feat = F.grid_sample(teacher_bev_feat, sampling_points)
            fit_loss = F.mse_loss(select_student_feat, select_teacher_feat)
            fit_loss = torch.mean(fit_loss)
            return dict(bev_fit_loss = fit_loss), None
        fg_map = None
        if gt_bboxes_list is not None and fg_pred_map is not None:
            raise Exception("distill fg weight should be specified!")
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = grid_size[:2] // 8
        if gt_bboxes_list is not None:
            device = student_bev_feat.device
            gt_bboxes_list = [torch.cat(
                    (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                    dim=1).to(device) for gt_bboxes in gt_bboxes_list]
            fg_map = student_bev_feat.new_zeros((len(gt_bboxes_list), feature_map_size[1], feature_map_size[0]))
            for idx in range(len(gt_bboxes_list)):
                num_objs = gt_bboxes_list[idx].shape[0]
                for k in range(num_objs):
                    width = gt_bboxes_list[idx][k][3]
                    length = gt_bboxes_list[idx][k][4]
                    width = width / voxel_size[0] / 8
                    length = length / voxel_size[1] / 8

                    if width > 0 and length > 0:
                        radius = gaussian_radius(
                            (length, width),
                            min_overlap=0.1)
                        radius = max(1, int(radius))

                        # be really careful for the coordinate system of
                        # your box annotation.
                        x, y, z = gt_bboxes_list[idx][k][0], gt_bboxes_list[idx][k][
                            1], gt_bboxes_list[idx][k][2]

                        coor_x = (
                            x - pc_range[0]
                        ) / voxel_size[0] / 8
                        coor_y = (
                            y - pc_range[1]
                        ) / voxel_size[1] / 8

                        center = torch.tensor([coor_x, coor_y],
                                            dtype=torch.float32,
                                            device=device)
                        center_int = center.to(torch.int32)

                        # throw out not in range objects to avoid out of array
                        # area when creating the heatmap
                        if not (0 <= center_int[0] < feature_map_size[0]
                                and 0 <= center_int[1] < feature_map_size[1]):
                            continue

                        draw_heatmap_gaussian(fg_map[idx], center_int, radius)
        if fg_pred_map is not None:
            fg_map = torch.max(fg_pred_map.sigmoid(), dim=1)[0]
        if fg_map is None:
            fg_map = student_bev_feat.new_ones((student_bev_feat.shape[0], feature_map_size[1], feature_map_size[0]))
        fit_loss = F.mse_loss(student_bev_feat, teacher_bev_feat, reduction='none')
        fit_loss = torch.sum(fit_loss * fg_map) / torch.sum(fg_map)
        return dict(bev_fit_loss = fit_loss), fg_map

    @force_fp32(apply_to=('student_bev_feat', 'teacher_bev_feat'))
    def get_heatmap_distill_loss(self, student_heatmap_pred, teacher_heatmap_pred, fg_map=None):
        num_task = len(student_heatmap_pred)
        kl_loss = 0
        for task_id in range(num_task):
            student_pred = student_heatmap_pred[task_id]['heatmap'].sigmoid()
            teacher_pred = teacher_heatmap_pred[task_id]['heatmap'].sigmoid()
            fg_map = fg_map.unsqueeze(1)
            task_kl_loss = F.binary_cross_entropy(student_pred, teacher_pred)
            task_kl_loss = torch.sum(task_kl_loss * fg_map) / torch.sum(fg_map)
            kl_loss += task_kl_loss
        return dict(
            heatmap_kl_loss = kl_loss
        )

    def convert_student_bev_feat(self, bev_feat):
        bev_num, batch_size, num_channels = bev_feat.shape
        bev_size = int(bev_num ** 0.5)
        bev_feat = bev_feat.permute(1, 2, 0).view(batch_size, num_channels, bev_size, bev_size)
        return bev_feat

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.student.forward_test(img_metas, imgs, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        print("I dont what to see this")
        exit()
        # since BEVFormer return pre_bev
        return self.student.simple_test(img_metas=img_metas, img=img, **kwargs)[1]

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(img_metas=img_metas, img=img, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)

    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data['img']
        img_metas = data['img_metas']
        img_feats = self.student.extract_feat(img=img,  img_metas=img_metas)
        prev_bev = data.get('prev_bev', None)
        prev_bev = self.student.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev

