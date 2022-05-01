# ML
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.ops import roi_align

# Code
from typing import Tuple, Optional
from numpy import ndarray
from torch import Tensor

# MMCV & MMDET
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import BitmapMasks

from mmdet.models.detectors import TwoStageDetector
from mmdet.models.builder import DETECTORS, HEADS, MODELS
from mmdet.models.dense_heads import RPNHead
from mmdet.models.roi_heads import BBoxHead, StandardRoIHead
from mmdet.models.backbones.resnet import Bottleneck
from mmdet.models.utils import ResLayer
from mmdet.core import encode_mask_results
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

# Model
from subprojects.sp02_omniiseg_fgn_mmdet.fgn_ag_rpn_head import AGRPNHead

# Stuff
from cp_utils.cp_time import datetime_now, datetime_diff
from cp_utils.cp_dir_file_ops import define_env, name_function


@HEADS.register_module()
class FGNBBoxHead(BBoxHead):
    n_ways = 3
    k_shots = 3

    def get_accuracy(self, cls_score: torch.Tensor, labels: torch.Tensor):
        final = torch.argmax(cls_score, dim=-1)
        assert len(cls_score) == len(labels)

        final = final.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        acc = dict()
        acc['ACC-Unbalanced'] = accuracy_score(y_true=labels, y_pred=final)
        acc['ACC-Balanced'] = balanced_accuracy_score(y_true=labels, y_pred=final)

        acc['ACC-Unbalanced'] = torch.Tensor([acc['ACC-Unbalanced']])
        acc['ACC-Balanced'] = torch.Tensor([acc['ACC-Balanced']])
        return acc

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                # if self.custom_activation:
                #     acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                #     losses.update(acc_)
                # else:
                #     losses['acc'] = accuracy(cls_score, labels)
                acc_ = self.get_accuracy(cls_score, labels)
                losses.update(acc_)
        if bbox_pred is not None:
            bg_class_ind = self.n_ways
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        """
        A single line changed is the background number
        MMDet now uses #num_classes for BG while for current FGN setup it is 2, 
        this simplifies the config and minimizes required architecture modifications.   
        When all branches (corresponding to few-shot values) are joined the background color 
        is equal to n_ways.      
        mmdet/models/roi_heads/bbox_heads/bbox_head.py
        """
        labels = pos_bboxes.new_full((num_samples,),
                                     self.n_ways,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        return super().get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat)

    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        return super().get_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale, cfg)


@MODELS.register_module()
class FGNRoIHead(StandardRoIHead):
    bbox_head: FGNBBoxHead
    n_ways = 3
    k_shots = 3
    subsampling_ratio = 16
    shared_head: ResLayer
    spp_fmaps_roi_aligned_cat_mean: Tensor
    # Mask Pool
    spp_fvecs_roi_aligned_cat_mean_mp: Tensor
    spp_vecs_mask: Tensor
    cls_reg_shared_conv: nn.Module
    cls_reg_shared_conv_norm: nn.Module
    fp16_enabled: bool = False
    teacher_forcing: bool = True

    def __init__(self, **kwargs):
        super(FGNRoIHead, self).__init__(**kwargs)
        self.init_shared_head()
        self.init_cls_reg_shared_conv()

    def init_shared_head(self):
        # True reverse engineering / hacking.
        # To omit downsampling and save the architecture I change the Bottleneck class params
        # and return them back after creation. Besides, I could have created another FGNBottleneck block class
        # and changed the ResNet.arch_settings[depth], but that is just another similar trick.
        Bottleneck.expansion = 2
        stage_block = 3
        # noinspection PyTypeChecker
        self.shared_head = ResLayer(
            block=Bottleneck,
            inplanes=1024,
            planes=512,
            num_blocks=stage_block,
            stride=1,
            dilation=1,
            style='pytorch',
            with_cp=False,
            # Possible normalization layers
            norm_cfg=dict(type='BN', requires_grad=True),
            dcn=None
        )
        # Initialization
        for m in self.shared_head.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        Bottleneck.expansion = 4
        return

    def shared_head_layer(self, x: Tensor) -> Tensor:
        out = self.shared_head(x)
        # print(name_function(), 'x: ', x.dtype, 'out: ', out.dtype)
        return out

    def init_cls_reg_shared_conv(self):
        self.cls_reg_shared_conv = nn.Conv2d(in_channels=2048, out_channels=1024,
                                             kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.cls_reg_shared_conv_norm = nn.GroupNorm(num_groups=32, num_channels=1024, affine=True)
        # self.cls_reg_shared_conv_norm = nn.BatchNorm2d(num_features=2048)

        # Convolutional layer
        nn.init.kaiming_normal_(self.cls_reg_shared_conv.weight, nonlinearity='relu')
        # nn.init.zeros(self.cls_reg_shared_conv.bias)
        # # Normalization layer
        nn.init.ones_(self.cls_reg_shared_conv_norm.weight)
        nn.init.zeros_(self.cls_reg_shared_conv_norm.bias)

    def count_one_roi_by_n_spp(self, bbox_feats: Tensor, rois):

        rois_amount = bbox_feats.shape[0]
        batch = self.spp_fmaps_roi_aligned_cat_mean.shape[0]
        # Not always this Sampler can sample self.bbox_sampler.num samples?
        # assert rois_amount / self.bbox_sampler.num == batch
        # [#ROI, 1024, 7, 7] -> [#ROI * N, 1024, 7, 7.]
        indexes = torch.repeat_interleave(torch.arange(len(rois)), self.n_ways).to(bbox_feats.device)
        c, h, w = bbox_feats.shape[-3:]
        rois_repeated = bbox_feats[indexes].view(rois_amount * self.n_ways, c, h, w)

        # [Batch, N, 1024, 7, 7]
        c, h, w = self.spp_fmaps_roi_aligned_cat_mean.shape[-3:]
        spps_repeated = self.spp_fmaps_roi_aligned_cat_mean.view(batch, self.n_ways, c, h, w)
        indexes = rois[:, 0].long()
        spps_repeated = spps_repeated[indexes].view(rois_amount * self.n_ways, c, h, w)

        one_roi_by_n_spp = torch.cat((rois_repeated, spps_repeated), dim=1)

        one_roi_by_n_spp_conv_out = self.cls_reg_shared_conv(one_roi_by_n_spp)
        one_roi_by_n_spp_norm_out = self.cls_reg_shared_conv_norm(one_roi_by_n_spp_conv_out)
        one_roi_by_n_spp_final = F.relu(one_roi_by_n_spp_norm_out)
        # print(name_function(),
        #       'bbox_feats: ', bbox_feats.dtype,
        #       'spp_fmaps_roi_aligned: ', spp_fmaps_roi_aligned.dtype,
        #       'one_roi_by_n_spp_final: ', one_roi_by_n_spp_final.dtype)
        return rois_amount, one_roi_by_n_spp_final

    def count_one_roi_by_n_spp_new(self, bbox_feats: Tensor, spp_fmaps_roi_aligned: Tensor):

        # [#ROI, 1024, 7, 7] <CONCAT> [N, 1024, 7, 7] = [#ROI * N, 2048, 7, 7]
        rois_amount = bbox_feats.shape[0]
        _, d, h, w = spp_fmaps_roi_aligned.shape
        spp_fmaps_aligned_grouped_n = spp_fmaps_roi_aligned.view(self.n_ways, self.k_shots, d, h, w)
        spp_fmaps_aligned_grouped_n_mean = spp_fmaps_aligned_grouped_n.mean(dim=1, keepdim=False)

        tensors = []
        for i in range(self.n_ways):
            single_spp_repeated = spp_fmaps_aligned_grouped_n_mean[i].repeat(repeats=(rois_amount, 1, 1, 1))
            one_roi_by_1_spp = torch.cat((bbox_feats, single_spp_repeated), dim=1)
            one_roi_by_1_spp_conv_out = self.cls_reg_shared_conv(one_roi_by_1_spp)
            one_roi_by_1_spp_norm_out = self.cls_reg_shared_conv_norm(one_roi_by_1_spp_conv_out)
            one_roi_by_1_spp_final = F.relu(one_roi_by_1_spp_norm_out, inplace=True)
            one_roi_by_1_spp_final = one_roi_by_1_spp_final.view(rois_amount, 1, 1024, 7, 7)
            tensors.append(one_roi_by_1_spp_final)
        one_roi_by_n_spp_final_new = torch.cat(tensors, dim=1).view(rois_amount * self.n_ways, 1024, 7, 7)

        return rois_amount, one_roi_by_n_spp_final_new

    def count_modified_cls_bbox(self, rois_amount, cls_score, bbox_pred):
        if self.n_ways == 1:
            cls_score_final = cls_score[:, [1, 0]]
            bbox_pred_final = bbox_pred
            return cls_score_final, bbox_pred_final
        else:
            assert self.n_ways == 3

        # [#ROI * N, 2]
        reshaped = cls_score.view(rois_amount, self.n_ways * 2)
        # Going from 1 since the 1st value indicates confidence in class
        # 1::2 = 1, 3, 5
        # Selection results in 0, 1, 2 -> * 2 = 0, 2, 4
        # (full correspondence to background-foreground labels)
        top = reshaped[:, [1, 3, 5]].argmax(dim=-1) * 2

        indexes = torch.arange(rois_amount)
        bg_class = reshaped[indexes, top].view(rois_amount, 1)
        pr_class = reshaped[:, [1, 3, 5]]
        # Improved in the last week before the deadline
        cls_score_final = torch.cat((pr_class, bg_class), dim=1)

        outputs_reg = bbox_pred.view(rois_amount, self.n_ways * 4)
        bbox_pred_final = outputs_reg
        return cls_score_final, bbox_pred_final

    def _bbox_forward(self, qry_fmap, rois):
        """Box head forward function used in both training and testing."""
        qry_fmap = torch.unsqueeze(qry_fmap, 0)
        bbox_feats = self.bbox_roi_extractor(
            qry_fmap[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head_layer(bbox_feats)

        rois_amount, one_roi_by_n_spp = self.count_one_roi_by_n_spp(bbox_feats, rois)

        cls_score_raw, bbox_pred_raw = self.bbox_head.forward(one_roi_by_n_spp)
        cls_score, bbox_pred = self.count_modified_cls_bbox(rois_amount, cls_score_raw, bbox_pred_raw)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward(self, qry_fmap, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            qry_fmap = torch.unsqueeze(qry_fmap, 0)
            mask_feats = self.mask_roi_extractor(
                qry_fmap[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            # Convert from torch.uint8 to torch.bool
            pos_inds_new = torch.nonzero(pos_inds).view(-1)
            mask_feats = bbox_feats[pos_inds_new]

        assert mask_feats.shape[:2] == self.spp_vecs_mask.shape[:2]
        assert mask_feats.shape[2:] == (7, 7)
        assert self.spp_vecs_mask.shape[2:] == (1, 1)
        mask_feats = mask_feats * self.spp_vecs_mask
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            assert self.teacher_forcing
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def count_spp(self, spp_fmaps, spp_bboxes, spp_isegmaps):
        """

        :param spp_fmaps: [Batch * N * K, C, H, W]
        :param spp_bboxes: [Batch * N * K, 1, 4] (For easier ROI Align)
        :param spp_isegmaps: [Batch * N * K,  1, H, W] (Binary)
        :return:
        """

        # noinspection PyTypeChecker
        spp_isegmaps_roi_aligned = roi_align(spp_isegmaps.float(), [s for s in spp_bboxes], output_size=7)
        spp_bboxes /= self.subsampling_ratio
        # noinspection PyTypeChecker
        spp_fmaps_roi_aligned = roi_align(spp_fmaps, [s for s in spp_bboxes], output_size=7)
        del spp_bboxes

        if self.with_shared_head:
            spp_fmaps_roi_aligned = self.shared_head_layer(spp_fmaps_roi_aligned)

        c, h, w = spp_fmaps_roi_aligned.shape[-3:]
        self.spp_fmaps_roi_aligned_cat_mean = spp_fmaps_roi_aligned \
            .view(-1, self.n_ways, self.k_shots, c, h, w) \
            .mean(dim=2) \
            .view(-1, self.n_ways, c, h, w)

        self.spp_fvecs_roi_aligned_cat_mean_mp = (spp_fmaps_roi_aligned * spp_isegmaps_roi_aligned) \
            .view(-1, self.n_ways, self.k_shots, c, h, w) \
            .mean(dim=(2, 4, 5)) \
            .view(-1, self.n_ways, c, 1, 1)

        return

    def forward_train(self,
                      qry_fmap,
                      img_metas,
                      proposal_list,
                      qry_bboxes,
                      qry_cat_ids,
                      qry_bboxes_ignore=None,
                      qry_isegmaps=None,
                      spp_fmaps=None,
                      spp_bboxes=None,
                      spp_isegmaps=None,
                      **kwargs):
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if qry_bboxes_ignore is None:
                qry_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], qry_bboxes[i], qry_bboxes_ignore[i],
                    qry_cat_ids[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    qry_bboxes[i],
                    qry_cat_ids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in qry_fmap])
                sampling_results.append(sampling_result)

        self.count_spp(spp_fmaps, spp_bboxes, spp_isegmaps)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                qry_fmap, sampling_results, qry_bboxes, qry_cat_ids, img_metas)
            losses.update(bbox_results['loss_bbox'])

        qry_isegmaps_bm = []
        for i in range(len(qry_isegmaps)):
            h, w = qry_isegmaps[i].shape[-2:]
            bm = BitmapMasks(qry_isegmaps[i].cpu().numpy(), height=h, width=w)
            qry_isegmaps_bm.append(bm)

        # mask head forward and loss
        if self.with_mask:
            # bbox_results['bbox_feats'] are RoI extracted with bboxes
            # which were passed through shared head if it exists
            mask_sampling_results = sampling_results
            # if self.teacher_forcing:
            #     mask_sampling_result = \
            #         type('MaskSamplingResult', (object,),
            #              {'neg_bboxes': qry_bboxes[0][:0],
            #               'pos_bboxes': qry_bboxes[0],
            #               'bboxes': qry_bboxes[0],
            #               'pos_gt_labels': qry_cat_ids[0],
            #               'pos_assigned_gt_inds': torch.arange(len(qry_bboxes[0])).cuda(),
            #               'pos_is_gt': torch.ones(len(qry_bboxes)).cuda()
            #               })
            #     mask_sampling_results = [mask_sampling_result]

            batch, n, c = self.spp_fvecs_roi_aligned_cat_mean_mp.shape[:3]
            assert batch == len(mask_sampling_results)
            assert n == self.n_ways

            indexes = [mask_sampling_results[i].pos_gt_labels + self.n_ways * i
                       for i in range(len(mask_sampling_results))]
            indexes = torch.cat(indexes)

            spp_vecs_all = self.spp_fvecs_roi_aligned_cat_mean_mp\
                .view(batch * self.n_ways, c, 1, 1)
            self.spp_vecs_mask = spp_vecs_all[indexes]

            # noinspection PyTypeChecker
            mask_results = self._mask_forward_train(
                qry_fmap, mask_sampling_results, bbox_results['bbox_feats'], qry_isegmaps_bm, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return segm_results

    def simple_test(self,
                    qry_fmap,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    spp_fmaps=None,
                    spp_bboxes=None,
                    spp_isegmaps=None):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        self.count_spp(spp_fmaps, spp_bboxes, spp_isegmaps)
        det_bboxes, det_labels = self.simple_test_bboxes(
            qry_fmap, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        """
        bbox_results = [
            # Change num_classes because in the result it is self.n_ways but not 1
            bbox2result(det_bboxes[i], det_labels[i], num_classes=self.n_ways)
            for i in range(len(det_bboxes))
        ]
        """

        # Works only for batch = 1
        # self.count_spp_fmaps_isegmaps_prod_mean_all(spp_bboxes, spp_isegmaps, indexes=det_labels[0])

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            # Because labels vary in range (0, self.n_ways), but there is a single mask,
            # addressing mask num 2 will lead to error. Feeding 0 label to address #0 mask.
            det_labels_gather = [det_labels[i] + self.n_ways * i for i in range(len(det_labels))]
            det_labels_gather = torch.cat(det_labels_gather)

            batch, n, c = self.spp_fvecs_roi_aligned_cat_mean_mp.shape[:3]
            assert batch == len(det_labels)
            assert n == self.n_ways
            spp_vecs_all = self.spp_fvecs_roi_aligned_cat_mean_mp.view(batch * self.n_ways, c, 1, 1)
            self.spp_vecs_mask = spp_vecs_all[det_labels_gather]

            det_labels_mask = [det_labels[i] * 0 for i in range(len(det_labels))]
            segm_results = self.simple_test_mask(
                qry_fmap, img_metas, det_bboxes, det_labels_mask, rescale=rescale)
            return det_bboxes, det_labels, segm_results
