import torch

# For faster and better types annotation
from typing import Optional

from torch import Tensor

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RPNHead
# BaseSampler, RandomSampler, OHEMSampler
from mmdet.core.bbox.samplers import RandomSampler


@HEADS.register_module()
class AGRPNHead(RPNHead):
    n_ways = 3
    k_shots = 3
    verbose = False
    sampler: RandomSampler
    subsampling_ratio = 16
    log_mode = False

    def __init__(self, **kwargs):
        super(AGRPNHead, self).__init__(**kwargs)

    def forward_single(self, qry_fmap: Tensor,
                       spp_fmaps: Optional[Tensor] = None,
                       qry_bboxes: Optional[Tensor] = None,
                       qry_cat_ids: Optional[Tensor] = None,
                       img_metas_cpu: Optional[list] = None,
                       train_mode=False,
                       log_mode=False):
        batch, c, x_h, x_w = qry_fmap.shape
        qry_fmap = qry_fmap[:, None, :, :, :]
        assert qry_fmap.shape == (batch, 1, c, x_h, x_w)

        c, h, w = spp_fmaps.shape[-3:]
        spp_fvecs_cat_mean = spp_fmaps \
            .view(batch, self.n_ways, self.k_shots, c, h, w) \
            .mean(axis=(2, 4, 5)) \
            .view(batch, self.n_ways, c, 1, 1)
        assert spp_fvecs_cat_mean.shape == (batch, self.n_ways, c, 1, 1)

        qry_fmap_mod = qry_fmap * spp_fvecs_cat_mean
        assert qry_fmap_mod.shape == (batch, self.n_ways, c, x_h, x_w)
        qry_fmap_mod = qry_fmap_mod.view(batch * self.n_ways, c, x_h, x_w)

        rpn_cls_score, rpn_bbox_pred = super(AGRPNHead, self).forward_single(qry_fmap_mod)
        if log_mode:
            self.qry_fmap_mod = qry_fmap_mod
            self.rpn_cls_score = rpn_cls_score
            self.rpn_bbox_pred = rpn_bbox_pred
            self.rpn_cls_score = rpn_cls_score

        rpn_losses: dict = {}

        assert train_mode ^ (qry_bboxes is None and qry_cat_ids is None)
        if train_mode:
            qry_bboxes_indexed_all = []
            img_metas_cpu_all = []
            for i in range(batch):
                for j in range(self.n_ways):
                    # print('N:', i)
                    indexes = torch.where(qry_cat_ids[i] == j)[0]
                    # print('Indexes', indexes)
                    qry_bboxes_indexed = qry_bboxes[i][:0]
                    if len(indexes) != 0:
                        qry_bboxes_indexed = qry_bboxes[i][indexes].view(-1, 4)
                    qry_bboxes_indexed_all.append(qry_bboxes_indexed)
                    img_metas_cpu_all.append(img_metas_cpu[i])

            assert len(img_metas_cpu_all) == batch * self.n_ways
            assert isinstance(img_metas_cpu_all[0], dict)
            loss_inputs = ([rpn_cls_score], [rpn_bbox_pred], qry_bboxes_indexed_all, img_metas_cpu_all)
            rpn_losses = self.loss(*loss_inputs)

            # Balancer is a little
            rpn_losses['loss_rpn_cls'][0] /= self.n_ways
            rpn_losses['loss_rpn_bbox'][0] /= self.n_ways

        # Change the last dim from 1 for sigmoid or 2 for softmax
        _, c, x_h, x_w = rpn_cls_score.shape
        rpn_cls_score = rpn_cls_score.view(batch, self.n_ways, c, x_h, x_w)
        _, c, x_h, x_w = rpn_bbox_pred.shape
        rpn_bbox_pred = rpn_bbox_pred.view(batch, self.n_ways, c, x_h, x_w)

        if self.n_ways > 1:
            rpn_cls_score_new_all = []
            rpn_bbox_pred_new_all = []
            for i in range(batch):
                a_scores = rpn_cls_score[i].permute(0, 2, 3, 1).reshape(self.n_ways, -1, 1)
                a_deltas = rpn_bbox_pred[i].permute(0, 2, 3, 1).reshape(self.n_ways, -1, 4)
                index = 1 if a_scores.shape[-1] == 2 else 0
                argmax = torch.argmax(a_scores[:, :, index], dim=0)
                arranged = torch.arange(len(argmax))

                rpn_cls_score_new = a_scores[argmax, arranged, :]
                rpn_bbox_pred_new = a_deltas[argmax, arranged, :]

                _, n_anchor_values_cls, h, w = rpn_cls_score[i].shape
                rpn_cls_score_new = rpn_cls_score_new.view(1, h, w, n_anchor_values_cls).permute(0, 3, 1, 2)
                rpn_cls_score_new_all.append(rpn_cls_score_new)
                _, n_anchor_values_bbox, h, w = rpn_bbox_pred[i].shape
                rpn_bbox_pred_new = rpn_bbox_pred_new.view(1, h, w, n_anchor_values_bbox).permute(0, 3, 1, 2)
                rpn_bbox_pred_new_all.append(rpn_bbox_pred_new)

            rpn_cls_score_new_all = torch.cat(rpn_cls_score_new_all, dim=0)
            rpn_bbox_pred_new_all = torch.cat(rpn_bbox_pred_new_all, dim=0)
        else:
            c, x_h, x_w = rpn_cls_score.shape[-3:]
            rpn_cls_score_new_all = rpn_cls_score.view(batch, c, x_h, x_w)
            c, x_h, x_w = rpn_bbox_pred.shape[-3:]
            rpn_bbox_pred_new_all = rpn_bbox_pred.view(batch, c, x_h, x_w)

        if train_mode:
            return rpn_cls_score_new_all, rpn_bbox_pred_new_all, rpn_losses

        return rpn_cls_score_new_all, rpn_bbox_pred_new_all
