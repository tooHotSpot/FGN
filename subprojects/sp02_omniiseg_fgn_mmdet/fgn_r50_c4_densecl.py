import numpy as np
from cp_utils.cp_dir_file_ops import define_env

checkpoint_fp = ''
cur_env = define_env()
if cur_env == 'PC':
    assert False
elif cur_env == 'SERVER':
    checkpoint_fp = '/home/neo/Downloads/densecl_r50_coco_1600ep.pth'
elif cur_env == 'COLAB':
    checkpoint_fp = '/content/drive/MyDrive/ColabNotebooks/densecl_r50_coco_1600ep.pth'

model = dict(
    type='FGN',
    backbone=dict(
        type='ResNet',
        # 18 and 34 use BasicBlocks while 50+ use BottleNeck blocks
        # This file is for 50 layers ONLY since consequent stages depend on depth
        depth=50,
        # Feed strides and dilations in the same amount as num_stages
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        # Output stage index, from 0 to 3 for default 4 blocks
        out_indices=(2,),
        # Replace 7x7 conv in input stem with 3 3x3 conv
        deep_stem=False,
        # For less information loss in Residual blocks
        avg_down=False,
        # To not freeze any layer, use -1
        frozen_stages=4,
        # Standard norm_layer is 32
        # norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        norm_cfg=dict(type='BN', requires_grad=False),
        # Does not concern non-BatchNorm layer
        norm_eval=True,
        # Again for lower information loss in Bottleneck blocks
        style='pytorch',
        init_cfg=[
            dict(type='Pretrained', checkpoint=checkpoint_fp)
        ]
    ),
    rpn_head=dict(
        type='AGRPNHead',
        num_convs=1,
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type='AnchorGenerator',
            # Anchors distribution are the same as in
            # mmdetection\configs\_base_\models\mask_rcnn_r50_caffe_c4.py
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
        init_cfg=[
            dict(type='Xavier', distribution='normal', layer='Conv2d'),
        ]
    ),
    roi_head=dict(
        type='FGNRoIHead',
        # Implemented internally
        shared_head=None,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            # type='ConvFCBBoxHead',
            type='FGNBBoxHead',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=1024,
            # in_channels=1024,
            # num_cls_convs=0,
            # num_reg_convs=0,
            # num_shared_convs=0,
            # conv_out_channels=1024,
            # fc_out_channels=1024,
            # num_cls_fcs=1,
            # num_reg_fcs=1,
            # roi_feat_size=7,
            # norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            # Num classes 1 results in cls_classes = num_classes + 1
            # But does not result in 2 regression layers due to implementation!
            num_classes=1,
            reg_class_agnostic=False,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            # For CrossEntropyLoss implementation in the mmdet num_classes specified for
            # in bbox_head config does not change anything since
            # loss_cls=dict(type='LabelSmoothingLoss', classes=4, smoothing=0.1),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
            init_cfg=[
                dict(type='Kaiming', distribution='normal', layer='Conv2d', nonlinearity='relu'),
                dict(type='Xavier', distribution='normal', layer='Linear')
            ],
        ),
        # For ResNet-C4 backbone mask_roi_extractor may be shared with the bbox_roi_extractor
        # But also it is required for teacher forcing
        # mask_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        #     out_channels=1024,
        #     featmap_strides=[16]),
        mask_head=dict(
            type='FCNMaskHead',
            # Prevent abnormal initialization in FCNMaskHead class
            init_cfg=None,
            # Set to 4 for better segmentation
            num_convs=4,
            in_channels=1024,
            conv_out_channels=256,
            num_classes=1,
            # Possible normalization layers, not implemented in the mmdet FCNHead
            # norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
            # norm_cfg=dict(type='BN', requires_grad=True),
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # For more frequent matching
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                # Lower for smaller AG-RPN, standard 256
                num=64,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                # Lower for smaller RG-Det Head, standard 512
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=14,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            nms=dict(type='nms', iou_threshold=0.7),
            # max_per_img=300 (Resnet-C4, Original article)
            # max_per_img=1000 (MMDet)
            max_per_img=300,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
