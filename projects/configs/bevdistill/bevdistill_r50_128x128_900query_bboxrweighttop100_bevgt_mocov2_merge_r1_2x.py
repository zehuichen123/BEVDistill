_base_ = [
    # '../datasets/custom_nus-3d.py',
    '../video_bevformer/bevformer_128x128_r50_2x_merge_r1_student.py',
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 128
bev_w_ = 128
queue_length = 4 # each sequence contains `queue_length` frames.

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
find_unused_parameters=True
distiller = dict(
    type='BEVDistiller',
    teacher_pretrained = '/nfs/chenzehui/code/bevformer/work_dirs/3detr_dv_fullset_2enc_aug_pretrain_reg05_heatmap_r1_merge_4subset/epoch_20.pth',  # mAP 64.5, NDS 70
    student_pretrained = 'data/pretrain_models/fcos3d_r50_new.pth',
    # student related params
    loss_cls_distill=dict(
        type='DistillCrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=0.0),
    loss_bbox_distill=dict(type='L1Loss', loss_weight=0.25),
    loss_iou_distill=dict(type='GIoULoss', loss_weight=0.0),
    distill_assigner=dict(
        type='DistillHungarianAssigner3D',
        cls_cost=dict(
            type='DistillCrossEntropyLossCost',
            # use_sigmoid=True,
            weight=1.0),
        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
        iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
        pc_range=point_cloud_range),
    # teacher related params
    distill_bbox_coder=dict(
        type='VanillaCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=50,
        voxel_size=voxel_size,
        num_classes=10), 
    score_reweight=True,
    loss_bev_distill=dict(
        type='L2Loss',
        loss_weight=0.01,
        fg_weight='gt'
    ),
    cl_head=dict(
        type='MoCoCLHead',
        img_channels=256,
        pts_channels=256,
        mid_channels=512,
        loss_cl=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ),
    center_aux_loss=True,
    use_heatmap_feat=False,
    distill_bbox=True,
)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    # load points
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    # load images
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # load annotations
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # parse images
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    # parse point clouds
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=6, pipeline=test_pipeline)
checkpoint_config = dict(interval=6)
find_unused_parameters=True
load_from = None
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

student_cfg = 'projects/configs/video_bevformer/bevformer_128x128_r50_2x_merge_r1_student.py'
teacher_cfg = 'projects/configs/3detr/3detr_dv_fullset_2enc_aug_pretrain_reg05_heatmap_r1_merge_4subset_teacher.py'