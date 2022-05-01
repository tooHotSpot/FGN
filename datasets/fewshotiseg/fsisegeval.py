import os
import numpy as np

from typing import Union
from collections import defaultdict

from pycocotools.cocoeval import COCOeval
from pycocotools.mask import area
from mmdet.core import encode_mask_results

from cp_utils.cp_dir_file_ops import check_dir_if_exists, check_file_if_exists, read_pkl


class FSISEGEval(COCOeval):
    def __init__(
            self,
            results_pkl_dir_fp: Union[str, None] = None,
            n_ways=3,
            iou_type='segm'
    ):
        """
        Modified class version

        :param imgs:
        :param gts: gt annotations with isegmaps
        :param dts: det annotation with isegmaps
        """
        assert results_pkl_dir_fp is not None
        super(FSISEGEval, self).__init__(cocoGt=None, cocoDt=None, iouType=iou_type)
        print('Evaluation (FSISEGEval) init ...')

        # Create a dict with imgs, gts, dts
        imgs = []
        gts = []
        dts = []
        # Just for debug purposes
        # https://github.com/cocodataset/cocoapi/issues/507#issuecomment-857272753
        gt_id_cum = 1
        dt_id_cum = 1

        total = 0
        if results_pkl_dir_fp is not None:
            results_files_all = sorted(os.listdir(results_pkl_dir_fp))
            assert len(results_files_all) != 0
            for file in results_files_all:
                result_file_fp = os.path.join(results_pkl_dir_fp, file)
                assert check_file_if_exists(result_file_fp)
                results = read_pkl(result_file_fp)
                # print('FSISEGEval Reading file', result_file_fp)
                # print('Total entries in file', len(results))
                for res in results:
                    # Do not forget zero
                    # if 'qry_isegmaps_rle' not in res:
                    #     res['qry_isegmaps_rle'] = encode_mask_results([res['qry_isegmaps']])[0]
                    # if 'dt_isegmaps_rle' not in res:
                    #     res['dt_isegmaps_rle'] = encode_mask_results([res['dt_isegmaps']])[0]

                    img_shape = res['qry_img_shape']
                    imgs.append({
                        'height': img_shape[0],
                        'width': img_shape[1]
                    })
                    bboxes = res['qry_bboxes']
                    res['qry_bboxes'] = np.column_stack((
                        bboxes[:, 1],
                        bboxes[:, 0],
                        np.maximum(bboxes[:, 3] - bboxes[:, 1], 1),
                        np.maximum(bboxes[:, 2] - bboxes[:, 0], 1)
                    ))
                    del bboxes
                    for j in range(len(res['qry_bboxes'])):
                        gts.append({
                            'image_id': total,
                            'id': gt_id_cum,
                            'bbox': res['qry_bboxes'][j],
                            'segmentation': res['qry_isegmaps_rle'][j],
                            'category_id': res['qry_cat_ids'][j],
                            'area': area(res['qry_isegmaps_rle'][j]),
                            'iscrowd': 0,
                            'ignore': False
                        })
                        gt_id_cum += 1
                    # res['dt_bboxes'] = res['dt_bboxes'][:, [1, 0, 3, 2]]
                    bboxes = res['dt_bboxes']
                    res['dt_bboxes'] = np.column_stack((
                        bboxes[:, 1],
                        bboxes[:, 0],
                        np.maximum(bboxes[:, 3] - bboxes[:, 1], 1),
                        np.maximum(bboxes[:, 2] - bboxes[:, 0], 1)
                    ))
                    del bboxes
                    for k in range(len(res['dt_bboxes'])):
                        dts.append({
                            'image_id': total,
                            'id': dt_id_cum,
                            'bbox': res['dt_bboxes'][k],
                            'segmentation': res['dt_isegmaps_rle'][k],
                            'category_id': res['dt_cat_ids'][k],
                            'score': res['dt_scores'][k],
                            'area': area(res['dt_isegmaps_rle'][k])
                        })
                        dt_id_cum += 1
                    # Increment
                    total += 1

        print('Total images', total)
        # region Change evaluation params
        self.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .10)) + 1, endpoint=True)
        self.params.iouThrs = [0.5]
        self.params.maxDets = [100]
        self.params.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.params.areaRngLbl = ['all']

        self.params.imgIds = np.arange(total)
        self.params.catIds = np.arange(n_ways)
        self.params.useCats = 1
        # endregion

        # imgs ids with <height> and <width> params
        assert imgs is not None
        self.imgs = imgs
        # Required to have <image_id>, <id>, <segmentation>, <category_id>, <area>, <iscrowd>, <ignore> keys
        assert gts is not None
        self.gts = gts
        # Required to have <image_id>, <id>, <segmentation>, <category_id>, <area>, <score> keys
        assert dts is not None
        self.dts = dts

    def _prepare(self):
        """
        This is a function imported from the pycocotools.COCOeval and modified

        *** Prepare ._gts and ._dts for non-COCO style
        dataset like OMNIISEG and MNISTISEG ***
        :return: None
        """

        for gt in self.gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']

        for gt in self.gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in self.dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        # per-image per-category evaluation results
        self.evalImgs = defaultdict(list)
        # accumulated evaluation results
        self.eval = {}

    def summarize_short(self, iouThr=0.5, aind=0, mind=0):
        areaRngLbl = self.params.areaRng[aind]
        maxDets = self.params.maxDets[mind]
        # Dimension of precision: [TxRxKxAxM]
        s = self.eval['precision']
        # Index is 0, computing it explicitly causes unknown errors
        # t = np.where(self.params.iouThrs == iouThr)[0]
        # s = s[t]
        s = s[0, :, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s_precision = 0
        else:
            mean_s_precision = np.mean(s[s > -1])

        # Dimension of recall: [TxKxAxM]
        s = self.eval['recall']
        # Index is 0, computing it explicitly causes unknown errors
        # t = np.where(self.params.iouThrs == iouThr)[0]
        # s = s[t]
        s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s_recall = 0
        else:
            mean_s_recall = np.mean(s[s > -1])

        print(
            f'IoU: {iouThr:0.3} '
            f'Area:{areaRngLbl} '
            f'MD:  {maxDets:3} '
            f'mAP: {mean_s_precision:3f} '
            f'mAR: {mean_s_recall:.3f} '
        )

        self.stats = [mean_s_precision, mean_s_recall]
        return {'mAP': mean_s_precision, 'mAR': mean_s_recall}
