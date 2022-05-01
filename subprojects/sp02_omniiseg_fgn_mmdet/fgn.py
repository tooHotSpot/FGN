import torch
import numpy as np

# For faster and better types annotation
from typing import Tuple, Optional, Dict
from numpy import ndarray
from torch import Tensor

from mmdet.models.detectors import TwoStageDetector
from mmdet.models.builder import DETECTORS

# NN
from datasets.fewshotiseg.base_fst import BaseFewShotISEG
from subprojects.sp02_omniiseg_fgn_mmdet.fgn_ag_rpn_head import AGRPNHead
from subprojects.sp02_omniiseg_fgn_mmdet.fgn_roi_head import FGNRoIHead

# Visualization
import cv2
import matplotlib.pyplot as plt
from imgaug.augmenters import Resize, Pad
from typing import List

# Stuff
from cp_utils.cp_dir_file_ops import define_env
from mmdet.core import encode_mask_results


@DETECTORS.register_module()
class FGN(TwoStageDetector):
    n_ways = 3
    k_shots = 3
    rpn_head: AGRPNHead
    roi_head: FGNRoIHead
    fp16_enabled: bool = False
    fp16_enable_on_train: bool = False
    teacher_forcing = True
    ds: Optional[BaseFewShotISEG] = None
    subsampling_ratio = 16
    frozen = False

    def __init__(self, n_ways, k_shots, **kwargs):
        super(FGN, self).__init__(**kwargs)
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.rpn_head.n_ways = n_ways
        self.rpn_head.k_shots = k_shots
        self.roi_head.n_ways = n_ways
        self.roi_head.k_shots = k_shots
        self.roi_head.bbox_head.n_ways = n_ways
        self.roi_head.bbox_head.k_shots = k_shots

        if self.backbone.frozen_stages != -1:
            self.frozen = True

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            # Base class implementation is a little bit complex
            return self.simple_test(**kwargs)

    def extract_feats(self, imgs):
        # Do not use this method since it leads
        # to longer pass through convolutional layers
        raise NotImplementedError

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if self.frozen:
            with torch.no_grad():
                x = self.backbone(img)
        else:
            x = self.backbone(img)

        # if self.with_neck:
        #     x = self.neck(x)
        return x

    @staticmethod
    def modify_input(qry_img: Tensor,
                     qry_bboxes: Tensor,
                     qry_cat_ids: Tensor,
                     qry_isegmaps: Tensor,
                     spp_imgs: Tensor,
                     spp_bboxes: Tensor,
                     spp_isegmaps: Tensor) -> Tuple[Tensor, ...]:
        """
        Change device if required. May not work on define_env() == PC option.
        """
        batch = len(qry_img)
        # Place vars to GPU if available
        if define_env() in ('SERVER', 'COLAB'):
            qry_img = qry_img.cuda(non_blocking=True)
            qry_bboxes = [qry_bboxes[i].cuda(non_blocking=True) for i in range(batch)]
            qry_cat_ids = [qry_cat_ids[i].cuda(non_blocking=True) for i in range(batch)]
            qry_isegmaps = [qry_isegmaps[i].cuda(non_blocking=True) for i in range(batch)]
            spp_imgs = spp_imgs.cuda(non_blocking=True)
            spp_bboxes = spp_bboxes.cuda(non_blocking=True)
            spp_isegmaps = spp_isegmaps.cuda(non_blocking=True)

        # Change bboxes format from YXYX to XYXY
        # qry_bboxes = qry_bboxes[:, [1, 0, 3, 2]]
        # spp_bboxes = spp_bboxes[:, [1, 0, 3, 2]]
        for i in range(len(qry_bboxes)):
            qry_bboxes[i] = qry_bboxes[i][:, [1, 0, 3, 2]]
            spp_bboxes[i] = spp_bboxes[i][:, [1, 0, 3, 2]]

        return qry_img, qry_bboxes, qry_cat_ids, qry_isegmaps, spp_imgs, spp_bboxes, spp_isegmaps

    @staticmethod
    def get_img_metas(img_shape):
        img_metas_cpu = []
        img_metas_cuda = []
        for i in range(len(img_shape)):
            # Create img_metas for RPN and ROIHead placed at different devices
            s = np.ones(4, dtype=np.float32)
            t = img_shape[i].cuda()
            e_cuda = [dict(num_samples=1, pad_shape=t, img_shape=t, ori_shape=t, scale_factor=s)]
            img_metas_cuda.extend(e_cuda)
            t = img_shape[i].cpu()
            e_cpu = [dict(num_samples=1, pad_shape=t, img_shape=t, ori_shape=t, scale_factor=s)]
            img_metas_cpu.extend(e_cpu)
        return img_metas_cpu, img_metas_cuda

    def forward_train(self,
                      qry_img,
                      qry_bboxes,
                      qry_cat_ids,
                      qry_isegmaps,
                      qry_bboxes_ignore=None,
                      proposals=None,
                      spp_imgs=None,
                      spp_bboxes=None,
                      spp_isegmaps=None,
                      img_shape=None,
                      **kwargs):
        # Place vars to GPU if available and change bboxes format from YXYX to XYXY format
        qry_img, qry_bboxes, qry_cat_ids, qry_isegmaps, spp_imgs, spp_bboxes, spp_isegmaps = \
            self.modify_input(qry_img, qry_bboxes, qry_cat_ids, qry_isegmaps, spp_imgs, spp_bboxes, spp_isegmaps)
        img_metas_cpu, img_metas_cuda = self.get_img_metas(img_shape=img_shape)

        qry_fmap = self.extract_feat(qry_img)[0]
        c, h, w = spp_imgs.shape[-3:]
        spp_imgs = spp_imgs.view(-1, c, h, w)
        spp_fmaps = self.extract_feat(spp_imgs)[0]
        spp_bboxes = spp_bboxes.view(-1, 1, 4)
        h, w = spp_isegmaps.shape[-2:]
        spp_isegmaps = spp_isegmaps.view(-1, 1, h, w)

        losses = dict()

        self.rpn_head.log_mode = True
        rpn_cls_score, rpn_bbox_pred, rpn_losses = \
            self.rpn_head.forward_single(qry_fmap,
                                         spp_fmaps,
                                         qry_bboxes=qry_bboxes,
                                         qry_cat_ids=qry_cat_ids,
                                         img_metas_cpu=img_metas_cpu,
                                         train_mode=True)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.train_cfg.rpn)
        proposal_list = self.rpn_head.get_bboxes(
            [rpn_cls_score],
            [rpn_bbox_pred],
            img_metas=img_metas_cpu,
            cfg=proposal_cfg
        )
        losses.update(rpn_losses)

        assert qry_bboxes[0].shape[-1] == 4 and len(qry_bboxes[0].shape) == 2
        roi_losses = self.roi_head.forward_train(
            qry_fmap,
            img_metas_cuda,
            proposal_list,
            qry_bboxes,
            qry_cat_ids,
            qry_bboxes_ignore,
            qry_isegmaps,
            spp_fmaps,
            spp_bboxes,
            spp_isegmaps,
        )
        losses.update(roi_losses)

        return losses

    @torch.no_grad()
    def simple_test(self,
                    qry_img,
                    qry_bboxes,
                    qry_cat_ids=None,
                    qry_isegmaps=None,
                    qry_bboxes_ignore=None,
                    spp_imgs=None,
                    spp_bboxes=None,
                    spp_isegmaps=None,
                    qry_child_idx=None,
                    img_shape=None,
                    rescale=False,
                    cats_ids_to_sample_real=None,
                    spp_insts_ids=None,
                    idx=None,
                    **kwargs):
        """Test without augmentation."""
        # Place vars to GPU if available and change bboxes format from YXYX to XYXY format
        qry_img, _, _, _, spp_imgs, spp_bboxes, spp_isegmaps = \
            self.modify_input(qry_img, qry_bboxes, qry_cat_ids, qry_isegmaps, spp_imgs, spp_bboxes, spp_isegmaps)
        # Create img_metas for RPN and ROIHead placed at different devices
        img_metas_cpu, _ = self.get_img_metas(img_shape=img_shape)

        with torch.no_grad():
            qry_fmap = self.extract_feat(qry_img)[0]
            c, h, w = spp_imgs.shape[-3:]
            spp_imgs = spp_imgs.view(-1, c, h, w)
            spp_fmaps = self.extract_feat(spp_imgs)[0]
            spp_bboxes = spp_bboxes.view(-1, 1, 4)
            h, w = spp_isegmaps.shape[-2:]
            spp_isegmaps = spp_isegmaps.view(-1, 1, h, w)

        self.rpn_head.log_mode = False
        rpn_cls_score, rpn_bbox_pred = self.rpn_head.forward_single(qry_fmap, spp_fmaps)

        # self.visualize_spp_fmaps(int(idx), spp_imgs, spp_bboxes, spp_fmaps)
        # self.visualize_qry_fmaps(int(idx), qry_img[0], qry_bboxes, qry_fmap, self.rpn_head.qry_fmap_mod)
        # self.visualize_cls_scores(self.rpn_head.rpn_cls_score, qry_img[0])

        # Method in BBoxTestMixin, implemented in the AnchorHead
        # Convert back to Tensor if error will occur
        proposal_cfg = self.test_cfg.get('rpn_proposal', self.test_cfg.rpn)
        proposal_list = self.rpn_head.get_bboxes(
            [rpn_cls_score],
            [rpn_bbox_pred],
            img_metas=img_metas_cpu,
            cfg=proposal_cfg
        )

        outputs_all = self.roi_head.simple_test(
            qry_fmap,
            proposal_list,
            img_metas_cpu,
            rescale=rescale,
            spp_fmaps=spp_fmaps,
            spp_bboxes=spp_bboxes,
            spp_isegmaps=spp_isegmaps
        )

        inputs_all: Dict[str, List] = {
            # For merging results
            'idx': idx,
            # Do not put qry_img arrays in the shared memory, they are too big.
            # 'qry_img': qry_img,
            'qry_bboxes': qry_bboxes,
            'qry_isegmaps': qry_isegmaps,
            'qry_img_shape': img_shape,
            'qry_cat_ids': qry_cat_ids,
            # Indexes to restore all the initial batch info
            'qry_child_idx': qry_child_idx,
            'cats_ids_to_sample_real': cats_ids_to_sample_real,
            'spp_insts_ids': spp_insts_ids,
        }

        outputs_new_all = []
        batch = qry_img.shape[0]
        for i in range(batch):
            dt_bboxes_scores = outputs_all[0][i]
            dt_cat_ids = outputs_all[1][i]
            dt_isegmaps = outputs_all[2][i][0]

            dt_cat_ids = dt_cat_ids.reshape(-1)
            dt_scores = dt_bboxes_scores[:, -1]
            dt_scores = dt_scores.reshape(-1)
            # Convert back from XYXY to YXYX, inputs are in the correct order
            dt_bboxes = dt_bboxes_scores[:, [1, 0, 3, 2]]
            dt_bboxes = dt_bboxes.reshape(-1, 4)

            # Create a new dict with all required attributes
            outputs_new_one = {
                'dt_scores': dt_scores,
                'dt_bboxes': dt_bboxes,
                'dt_cat_ids': dt_cat_ids,
                'dt_isegmaps_rle': encode_mask_results([dt_isegmaps])[0]
            }
            # Do not use a self.modify_input but open each item
            for key in inputs_all.keys():
                outputs_new_one[key] = inputs_all[key][i]

            for key, value in outputs_new_one.items():
                if isinstance(value, Tensor):
                    if value.device != 'cpu':
                        value = value.cpu()
                    outputs_new_one[key] = value.numpy()

            # Minimize the result dict size with conversion to lighter data types
            # whole_outputs['qry_bboxes'] = np.around(whole_outputs['qry_bboxes']).astype(np.int16)
            # whole_outputs['qry_cat_ids'] = whole_outputs['qry_cat_ids'].astype(np.int8)
            # whole_outputs['dt_bboxes'] = np.around(whole_outputs['dt_bboxes']).astype(np.int16)
            # whole_outputs['dt_cat_ids'] = whole_outputs['dt_cat_ids'].astype(np.int8)
            encoded = encode_mask_results([outputs_new_one['qry_isegmaps']])[0]
            outputs_new_one['qry_isegmaps_rle'] = encoded
            outputs_new_one.pop('qry_isegmaps')

            outputs_new_all.append(outputs_new_one)
        return outputs_new_all

    def visualize_spp_fmaps(self, idx, spp_imgs: Tensor, spp_bboxes: Tensor, spp_fmaps: Tensor):
        # In total from 1024 to 2048 layers, selecting 8, first of each group
        spp_fmaps = spp_fmaps.detach()
        spp_imgs = spp_imgs.detach().cpu().numpy()
        spp_bboxes = spp_bboxes.detach().cpu().numpy()
        spp_bboxes = spp_bboxes[:, [1, 0, 3, 2]]

        assert spp_fmaps.ndim == 4
        assert len(spp_fmaps) == self.n_ways * self.k_shots
        _, c, h, w = spp_fmaps.shape
        assert h == w
        groups = 8
        channels_selected = np.arange(groups) * (c // groups)
        spp_fmaps = spp_fmaps[:, channels_selected, :, :]
        spp_fmaps_mean = torch.mean(spp_fmaps, dim=(2, 3), keepdim=True)
        spp_fmaps_std = torch.std(spp_fmaps, dim=(2, 3), keepdim=True)
        spp_fmaps_imgs = (spp_fmaps - spp_fmaps_mean) / spp_fmaps_std
        spp_fmaps_imgs *= 64
        spp_fmaps_imgs += 128
        spp_fmaps_imgs = torch.clip(spp_fmaps_imgs, 0, 255).to(torch.uint8)

        # *** Magic ***
        # [N * K, 8, H, W] -> [H, W, N * K, 8]
        h, w = spp_fmaps_imgs.shape[-2:]
        spp_fmaps_imgs_np: ndarray = spp_fmaps_imgs.cpu().numpy().transpose((2, 3, 0, 1))
        # [H, W, N * K, 8] -> [H, W, N * K * 8]
        spp_fmaps_imgs_np_depth_fwd = spp_fmaps_imgs_np.reshape((h, w, -1))
        spp_fmaps_imgs_np_resized = cv2.resize(spp_fmaps_imgs_np_depth_fwd, (128, 128), interpolation=cv2.INTER_CUBIC)
        pad = 3
        pad_value = 0
        spp_fmaps_imgs_np_resized[:pad, :] = pad_value
        spp_fmaps_imgs_np_resized[-pad:, :] = pad_value
        spp_fmaps_imgs_np_resized[:, :pad] = pad_value
        spp_fmaps_imgs_np_resized[:, -pad:] = pad_value
        # [H, W, N * K, 8] <- [H, W, N * K * 8]
        spp_fmaps_imgs_np_depth_bwd = spp_fmaps_imgs_np_resized.reshape(128, 128, self.n_ways * self.k_shots, groups)
        # [N * K, 8, H, W] <- [H, W, N * K, 8]
        spp_fmaps_imgs_np = spp_fmaps_imgs_np_depth_bwd.transpose(2, 3, 0, 1)

        spp_fmaps_rows = [np.column_stack(spp_fmaps_imgs_np[i])
                          for i in range(self.n_ways * self.k_shots)]
        spp_fmaps_grid = np.row_stack(spp_fmaps_rows)

        combined: List[ndarray] = []
        for i in range(self.n_ways * self.k_shots):
            spp_img_now = self.ds.denormalize(spp_imgs[i])
            spp_bbox_now = spp_bboxes[i]
            combined_now = self.ds.draw_on_img(spp_img_now, spp_bbox_now, cat_ids=[i // self.n_ways])
            combined_now = cv2.resize(combined_now, (128, 128), interpolation=cv2.INTER_CUBIC)
            combined_now[:pad, :] = 0
            combined_now[-pad:, :] = 0
            combined_now[:, :pad] = 0
            combined_now[:, -pad:] = 0
            combined.append(combined_now)
        spp_imgs_cols = np.row_stack(combined)
        spp_imgs_grid = np.column_stack([spp_imgs_cols for _ in range(groups)])

        alpha = 0.5
        overlayed = spp_imgs_grid * alpha + spp_fmaps_grid[..., np.newaxis] * (1 - alpha)
        overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

        final = np.column_stack([spp_imgs_cols, overlayed])

        # print('AGRPNHead: finished spp_fmaps plot')
        plt.imsave(f'imgs/SPPFM_{idx:05}.png', arr=final)

        return final

    def visualize_qry_fmaps(self, idx, qry_img: Tensor, qry_bboxes: Tensor, qry_fmap: Tensor, qry_fmaps_mod: Tensor):

        # In total from 1024 to 2048 layers, selecting 8, first of each group
        qry_img = qry_img.detach().cpu()
        qry_fmap = qry_fmap.detach().cpu()
        qry_fmaps_mod = qry_fmaps_mod.detach().cpu()
        qry_fmaps = torch.cat([qry_fmap, qry_fmaps_mod], dim=0)

        print(qry_bboxes)
        qry_bboxes = qry_bboxes.detach().cpu().numpy().reshape(-1, 4)
        qry_bboxes = qry_bboxes[:, [1, 0, 3, 2]]
        print(qry_bboxes)

        assert qry_fmaps.ndim == 4
        assert len(qry_fmaps_mod) == self.n_ways
        _, c, h, w = qry_fmaps.shape
        groups = 8
        channels_selected = np.arange(groups) * (c // groups)
        qry_fmaps = qry_fmaps[:, channels_selected, :, :]
        qry_fmaps[1:] = (qry_fmaps[0] - qry_fmaps[1:]).abs()
        # qry_fmaps_mean = torch.mean(qry_fmaps, dim=(2, 3), keepdim=True)
        # qry_fmaps_std = torch.std(qry_fmaps, dim=(2, 3), keepdim=True)
        # qry_fmaps_imgs = (qry_fmaps - qry_fmaps_mean) / qry_fmaps_std
        # qry_fmaps_imgs *= 64
        # qry_fmaps_imgs += 128
        qry_fmaps_min = torch.from_numpy(np.min(qry_fmaps.numpy(), axis=(2, 3), keepdims=True))
        qry_fmaps_max = torch.from_numpy(np.max(qry_fmaps.numpy(), axis=(2, 3), keepdims=True))
        qry_fmaps_imgs = (qry_fmaps - qry_fmaps_min) / (qry_fmaps_max - qry_fmaps_min)
        qry_fmaps_imgs = qry_fmaps_imgs * 255
        qry_fmaps_imgs = torch.clip(qry_fmaps_imgs, 0, 255).to(torch.uint8)

        n, _, h, w = qry_fmaps_imgs.shape
        qry_fmaps_imgs_np = qry_fmaps_imgs.cpu().numpy().reshape(n * groups, h, w)

        # [N, 8, H, W] -> [N * 8, H, W]
        from imgaug import augmenters as iaa
        res = iaa.Resize({'height': 256, 'width': 'keep-aspect-ratio'})
        pad = iaa.Pad(px=3, pad_mode='constant', pad_cval=0)
        qry_fmaps_imgs_all = res(images=qry_fmaps_imgs_np)
        h_new, w_new = qry_fmaps_imgs_all.shape[-2:]
        qry_fmaps_imgs_all = pad(images=qry_fmaps_imgs_all)
        h_pad, w_pad = qry_fmaps_imgs_all.shape[-2:]
        qry_fmaps_imgs_grouped = qry_fmaps_imgs_all.reshape(n, groups, h_pad, w_pad)
        rows = [np.column_stack(row) for row in qry_fmaps_imgs_grouped]
        grid = np.row_stack(rows)

        qry_img_np = self.ds.denormalize(qry_img)
        res_qry = iaa.Resize({'height': h_new, 'width': w_new})
        bbs = BaseFewShotISEG.get_bboxes_on_img_from_yxyx(qry_img, qry_bboxes)
        qry_img_res, bbs_res = res_qry(image=qry_img_np, bounding_boxes=bbs)
        qry_img_pad, bbs_pad = pad(image=qry_img_res, bounding_boxes=bbs_res)
        self.ds.draw_on_img(qry_img_pad, bboxes=bbs_pad)

        qry_img_column = np.row_stack([qry_img_pad] * n)
        qry_img_grid = np.column_stack([qry_img_column] * groups)

        alpha = 0.25
        grid_final = qry_img_grid * alpha + np.expand_dims(grid, -1) * (1 - alpha)
        grid_final = np.clip(grid_final, 0, 255).astype(np.uint8)
        final = np.column_stack((qry_img_column, grid_final))

        # print('AGRPNHead: finished qry fmap plot')
        plt.imsave(f'imgs/{idx}_QryFM.png', arr=final)

        return final

    def visualize_cls_scores(self, idx, rpn_cls_score: Tensor, qry_img: Tensor):
        scores = rpn_cls_score.detach()
        scores = torch.sigmoid(scores)
        assert scores.min() > -1.05
        assert scores.max() < 1.05
        scores = ((scores + 1) * 128)
        scores = torch.clamp(scores, min=0, max=255)
        scores = scores.to(torch.uint8).cpu().numpy()
        n, c, h, w = scores.shape
        scores = scores.reshape(-1, h, w)

        # About 64
        res = Resize(size={'height': 64, 'width': 'keep-aspect-ratio'})
        pad = Pad(px=3, pad_mode='constant', pad_cval=0)
        scores_res = res(images=scores)
        scores_pad = pad(images=scores_res)
        h, w = scores_pad.shape[-2:]
        scores_pad = scores_pad.reshape(n, c, h, w)

        qry_img = self.ds.denormalize(qry_img)
        res = Resize(size={'height': h, 'width': w})
        qry_img_res = res(image=qry_img)
        qry_img_pad = pad(image=qry_img_res)

        scores_bin_rows = [np.column_stack(scores_pad[i]) for i in range(n)]
        scores_bin_grid = np.row_stack(scores_bin_rows)
        scores_bin_grid = np.dstack([scores_bin_grid] * 3)

        scores_rgb = scores_pad.transpose((1, 2, 3, 0)).reshape(c, h, w, n)
        scores_rgb_row_rgb = np.column_stack(scores_rgb)

        qry_img_row_rgb = np.column_stack([qry_img_pad] * c)

        combined_row_rgb = qry_img_row_rgb * 0.25 + scores_rgb_row_rgb * 0.75
        combined_row_rgb = np.clip(combined_row_rgb, 0, 255).astype(np.uint8)

        whole = np.row_stack((scores_bin_grid, scores_rgb_row_rgb, combined_row_rgb))
        plt.imsave(f'imgs/{idx:03}_Scores_All.png', whole)
        return
