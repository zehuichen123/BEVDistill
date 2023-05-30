_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/schedules/cyclic_20e.py', 
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='ObjDGCNNDV',
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=-1, voxel_size=voxel_size, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        num_outs=4),
    pts_bbox_head=dict(
        type='DGCNN3DDistillHead',
        num_query=300,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        # max_radius=1,
        # share_conv_channel=256,
        # tasks=[
        #     dict(num_class=10, class_names=class_names),
        # ],
        transformer=dict(
            type='DeformableDetrTransformerV2',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=2,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='CustomMSDeformableAttentionFP16', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='Deformable3DDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        # dict(
                        #     type='DGCNNAttn',
                        #     embed_dims=256,
                        #     num_heads=8,
                        #     K=16,
                        #     dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttentionFP16',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.5),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)), # For DETR compatibility. 
     # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[1024, 1024, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=8,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.5),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=True,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=100)
    ))

# dataset_type = 'NuScenesDataset'
# data_root = 'data/nuscenes/'

# file_client_args = dict(backend='disk')

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'nuscenes_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(
#             car=5,
#             truck=5,
#             bus=5,
#             trailer=5,
#             construction_vehicle=5,
#             traffic_cone=5,
#             barrier=5,
#             motorcycle=5,
#             bicycle=5,
#             pedestrian=5)),
#     classes=class_names,
#     sample_groups=dict(
#         car=2,
#         truck=3,
#         construction_vehicle=7,
#         bus=4,
#         trailer=6,
#         barrier=2,
#         motorcycle=6,
#         bicycle=6,
#         pedestrian=2,
#         traffic_cone=2),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args))

# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=9,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args,
#         pad_empty_sweeps=True,
#         remove_close=True),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     # dict(
#     #     type='GlobalRotScaleTrans',
#     #     rot_range=[-0.3925, 0.3925],
#     #     scale_ratio_range=[0.85, 1.15],
#     #     translation_std=[0.5, 0.5, 0.5]),
#     # dict(
#     #     type='RandomFlip3D',
#     #     sync_2d=False,
#     #     flip_ratio_bev_horizontal=0.5,
#     #     flip_ratio_bev_vertical=0.5),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectNameFilter', classes=class_names),
#     dict(type='PointShuffle'),
#     dict(type='DefaultFormatBundle3D', class_names=class_names),
#     dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=9,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args,
#         pad_empty_sweeps=True,
#         remove_close=True),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1.0,
#         # flip=True,
#         # pcd_horizontal_flip=True,
#         # pcd_vertical_flip=True,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D', sync_2d=False),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#             dict(
#                 type='DefaultFormatBundle3D',
#                 class_names=class_names,
#                 with_label=False),
#             dict(type='Collect3D', keys=['points'])
#         ])
# ]

# # construct a pipeline for data and gt loading in show function
# # please keep its loading function consistent with test_pipeline (e.g. client)
# eval_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=9,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args,
#         pad_empty_sweeps=True,
#         remove_close=True),
#     dict(
#         type='DefaultFormatBundle3D',
#         class_names=class_names,
#         with_label=False),
#     dict(type='Collect3D', keys=['points'])
# ]

# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=6,
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file=data_root + 'nuscenes_infos_train.pkl',
#             pipeline=train_pipeline,
#             classes=class_names,
#             modality=input_modality,
#             test_mode=False,
#             use_valid_flag=True,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR',
#             load_interval=4),
#     ),
#     val=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality),
#     test=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality))

# evaluation = dict(interval=2, pipeline=eval_pipeline, start=20)

# runner = dict(type='EpochBasedRunner', max_epochs=20)
# optimizer = dict(
#     type='AdamW', 
#     lr=1e-4,
#     paramwise_cfg=dict(
#         custom_keys={
#             'pts_voxel_encoder': dict(lr_mult=0.1),
#             'SECOND': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# load_from='/nfs/chenzehui/code/autoalign/work_dirs/centerpoint_voxel_nus_fullset_dv_aug/epoch_20.pth'
# find_unused_parameters = True
# checkpoint_config = dict(interval=5)

# fp16 = dict(loss_scale=512.)