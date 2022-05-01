import os
import shutil

from tqdm import tqdm
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import matplotlib.pyplot as plt
from imgaug import BoundingBox, BoundingBoxesOnImage

import imagesize
from numpy import ndarray
from typing import List, Tuple, Union, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import ops

from pycocotools import mask

from cp_utils.cp_dir_file_ops import define_env, debugger_is_active
from cp_utils.cp_dir_file_ops import check_file_if_exists, check_dir_if_exists
from cp_utils.cp_dir_file_ops import create_empty_dir_safe
from cp_utils.cp_dir_file_ops import write_pkl_safe, read_pkl
from cp_utils.cp_dir_file_ops import write_json_safe, read_json

from datasets.voc.resources import Colors

if define_env() == 'PC':
    ds_root_path = 'C:/Users/Art/PycharmProjects/Datasets/VOC2012'
    # For fast data loading
    cv2.ocl.setUseOpenCL(False)
elif define_env() == 'SERVER':
    ds_root_path = '/home/neo/Datasets/VOC2012'
elif define_env() == 'COLAB':
    ds_root_path = '/content/VOC2012'
else:
    raise NotImplementedError('Paths are not specified')


def collate_fn_new(batch):
    return batch


num_workers = 0 if debugger_is_active() else 2
persistent_workers = not debugger_is_active()


class VOCDSParse(Dataset):
    root: str = '../../datasets/voc/resources'
    img_set: str
    ann_dir_fp: str
    imgs_dir_fp: str
    obj_isegmaps_dir_fp: str
    cls_isegmaps_dir_fp: str

    imgs_ids: List[str] = []
    ann_data: List[Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]] = []
    img_data: List[Tuple[ndarray, ndarray, ndarray, ndarray]] = []
    thres_iou: float = 0.25
    min_size_ratio: float = 0.005
    filter_min_size_ratio: bool = False

    verbose: bool = False
    excluded: List[str] = []

    voc_labels_coco_cats_ids: Dict[str, int] = {}
    voc_codes_coco_cats_ids: Dict[int, int] = {}

    def __init__(self, img_set):
        super(VOCDSParse, self).__init__()

        if not check_dir_if_exists(self.root):
            create_empty_dir_safe(self.root)
            assert check_dir_if_exists(self.root)
            print('Created dir', self.root)

        assert img_set in ('train', 'trainval', 'val')
        self.img_set = img_set

        self.ann_dir_fp = os.path.join(ds_root_path, 'Annotations')
        self.imgs_dir_fp = os.path.join(ds_root_path, 'JPEGImages')
        self.obj_isegmaps_dir_fp = os.path.join(ds_root_path, 'SegmentationObject')
        self.cls_isegmaps_dir_fp = os.path.join(ds_root_path, 'SegmentationClass')

        for dir_fp in (self.ann_dir_fp,
                       self.imgs_dir_fp,
                       self.obj_isegmaps_dir_fp,
                       self.cls_isegmaps_dir_fp):
            assert check_dir_if_exists(dir_fp)
            print('Checked DIR', dir_fp)

        # There are 4 subsets: Action, Layout, Main and Segmentation
        # ~1.5K images in SegmentationObject (for instance / semantic segmentation) and
        # ~17K in Main (for object detection)
        imgs_set_dir_fp = os.path.join(ds_root_path, 'ImageSets', 'Segmentation')
        assert check_dir_if_exists(imgs_set_dir_fp)

        ann_file_fp = os.path.join(imgs_set_dir_fp, f'{img_set}.txt')
        with open(ann_file_fp, mode='r') as f:
            self.imgs_ids = f.read().splitlines()

        print('Total images in the dataset to parse', len(self.imgs_ids))
        self.get_data()
        print('Initialization OK')
        return

    @staticmethod
    def get_bbox_from_mask_color(img_mask: ndarray, color: ndarray) -> ndarray:
        assert img_mask.ndim == 3
        assert color.ndim == 1 and len(color) == 3
        mask = np.array((img_mask[:, :, 0] == color[0]) *
                        (img_mask[:, :, 1] == color[1]) *
                        (img_mask[:, :, 2] == color[2])).astype(np.uint8) * 255
        select_y = np.max(mask, axis=1)
        select_x = np.max(mask, axis=0)
        select_y_indexes = np.where(select_y > 0)[0]
        select_x_indexes = np.where(select_x > 0)[0]
        ymin, ymax = select_y_indexes[[0, -1]]
        xmin, xmax = select_x_indexes[[0, -1]]
        bbox = np.array([ymin, xmin, ymax, xmax], dtype=np.int32)
        return bbox

    @staticmethod
    def get_bbox_from_mask_color_new(img_mask: ndarray, color: ndarray) -> ndarray:
        assert img_mask.ndim == 3
        assert color.ndim == 1 and len(color) == 3
        mask = np.min(img_mask == color, axis=-1)
        y, x = np.nonzero(mask)
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)
        bbox = np.array([ymin, xmin, ymax, xmax], dtype=np.int32)
        return bbox

    def get_ann_data_entry(self, idx):
        img_id: str = self.imgs_ids[idx]

        ann_file_fp: str = os.path.join(self.ann_dir_fp, img_id + '.xml')
        assert check_file_if_exists(ann_file_fp)

        tree = ET.parse(ann_file_fp)
        root = tree.getroot()

        bboxes: List[List[int]] = []
        cats_codes: List[int] = []
        cats_names: List[str] = []
        difficults: List[int] = []
        size: List[int] = []

        for child in root:
            if child.tag == 'object':
                bbox = child.find('bndbox')
                xmin: int = int(bbox.findtext('xmin'))
                ymin: int = int(bbox.findtext('ymin'))
                xmax: int = int(bbox.findtext('xmax'))
                ymax: int = int(bbox.findtext('ymax'))
                del bbox
                bbox = [ymin, xmin, ymax, xmax]
                bboxes.append(bbox)
                del bbox

                cat_name: str = child.findtext('name')
                # Remove the space in category name
                cat_name = cat_name.replace(' ', '')
                cats_names.append(cat_name)

                cat_code = Colors.voc_labels_codes[cat_name]
                cats_codes.append(cat_code)
                del cat_name, cat_code

                difficult = child.findtext('difficult')
                difficult = int(difficult)
                difficults.append(difficult)
                del difficult

            elif child.tag == 'size':
                # Change values
                h: int = int(child.findtext('height'))
                w: int = int(child.findtext('width'))
                d: int = int(child.findtext('depth'))
                if d == 1:
                    print(f'Single depth {d} for {img_id}')
                size = [h, w]
                del h, w

        # Parsed one data
        assert len(bboxes) == len(cats_names) == len(difficults)
        assert size is not None

        bboxes: ndarray = np.array(bboxes, dtype=np.int32).reshape(-1, 4)
        cats_codes: ndarray = np.array(cats_codes, dtype=np.int32).reshape(-1)
        cats_names: ndarray = np.array(cats_names, dtype=str).reshape(-1)
        difficults: ndarray = np.array(difficults, dtype=np.int32).reshape(-1)
        size: ndarray = np.array(size, dtype=np.int32).reshape(-1)

        return bboxes, cats_codes, cats_names, difficults, size

    def get_img_data_entry(self, idx, size=None):
        img_id = self.imgs_ids[idx]

        img_file_fp: str = os.path.join(self.imgs_dir_fp, img_id + '.jpg')
        obj_isegmap_file_fp: str = os.path.join(self.obj_isegmaps_dir_fp, img_id + '.png')
        cls_isegmap_file_fp: str = os.path.join(self.cls_isegmaps_dir_fp, img_id + '.png')

        for file_fp in (img_file_fp, obj_isegmap_file_fp, cls_isegmap_file_fp):
            # Check path
            assert check_file_if_exists(file_fp), f'File {file_fp} does not exist'

        if size is not None:
            h, w = size
            del size
            # Check size
            img_size_w, img_size_h = imagesize.get(img_file_fp)
            if img_size_h != h or img_size_w != w:
                print(f'Invalid HWD {(h, w)} / {img_size_h, img_size_w} for {img_id}')
                assert False

        obj_isegmap = cv2.imread(obj_isegmap_file_fp, cv2.IMREAD_COLOR)[..., ::-1]
        cls_isegmap = cv2.imread(cls_isegmap_file_fp, cv2.IMREAD_COLOR)[..., ::-1]

        # Select unique colors and the bounding box for each of them
        obj_colors_unique: ndarray = np.unique(obj_isegmap.reshape(-1, 3), axis=0)
        # cls_colors_unique: ndarray = np.unique(cls_isegmap.reshape(-1, 3), axis=0)

        bboxes: List[ndarray] = []
        cats_codes: List[int] = []
        cats_names: List[str] = []
        obj_colors: List[ndarray] = []

        for obj_color in obj_colors_unique:
            if tuple(obj_color) in (Colors.voc_background_color, Colors.voc_ignore_label_color):
                continue
            bbox = self.get_bbox_from_mask_color_new(obj_isegmap, obj_color)
            bboxes.append(bbox)
            del bbox

            # Create a binary mask and get the class of the mask
            obj_mask = np.min(obj_isegmap == obj_color, axis=-1, keepdims=True) * 1
            obj_cls_isegmap = obj_mask * cls_isegmap
            obj_cls_colors = np.unique(obj_cls_isegmap.reshape(-1, 3), axis=0)

            obj_cls_color = None
            for c in obj_cls_colors:
                c = tuple(c)
                if c in (Colors.voc_background_color, Colors.voc_ignore_label_color):
                    continue
                assert obj_cls_color is None
                obj_cls_color = c

            cat_name = Colors.voc_colors_labels[obj_cls_color]
            cat_name = cat_name.replace(' ', '')
            cats_names.append(cat_name)

            cat_code = Colors.voc_labels_codes[cat_name]
            cats_codes.append(cat_code)
            del obj_cls_color, cat_name, cat_code

            obj_colors.append(obj_color)

        bboxes: ndarray = np.array(bboxes, dtype=np.int32).reshape(-1, 4)
        cats_codes: ndarray = np.array(cats_codes, dtype=np.int32).reshape(-1)
        cats_names: ndarray = np.array(cats_names, dtype=str).reshape(-1)
        obj_colors: ndarray = np.array(obj_colors, dtype=np.int32).reshape(-1, 3)

        assert len(bboxes) == len(cats_codes) == len(cats_names) == len(obj_colors)
        return bboxes, cats_codes, cats_names, obj_colors

    def __len__(self):
        return len(self.imgs_ids)

    def __getitem__(self, idx):
        if len(self.ann_data) == 0 or len(self.img_data) == 0:
            d1 = self.get_ann_data_entry(idx)
            size = d1[-1]
            d2 = self.get_img_data_entry(idx, size=size)
            return d1, d2

        return self.get_result_entry(idx)

    def get_data(self):
        name = str.upper(self.img_set)
        ann_data_path = os.path.join(self.root, f'ANN_DATA_{name}.pkl')
        img_data_path = os.path.join(self.root, f'IMG_DATA_{name}.pkl')
        del name

        if not os.path.exists(ann_data_path) or not os.path.exists(img_data_path):
            ann_data: List[Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]] = []
            img_data: List[Tuple[ndarray, ndarray, ndarray, ndarray]] = []

            dl = DataLoader(self, batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn_new, persistent_workers=persistent_workers)
            for data in tqdm(dl, 'DATA LOOP', ncols=100):
                bboxes: ndarray
                cats_codes: ndarray
                cats_names: ndarray
                difficults: ndarray
                size: ndarray
                bboxes, cats_codes, cats_names, difficults, size = data[0][0]
                ann_data.append((bboxes, cats_codes, cats_names, difficults, size))

                bboxes: ndarray
                cats_codes: ndarray
                cats_names: ndarray
                obj_colors: ndarray
                bboxes, cats_codes, cats_names, obj_colors = data[0][1]
                img_data.append((bboxes, cats_codes, cats_names, obj_colors))

            write_pkl_safe(ann_data_path, ann_data)
            print('Wrote data to file: ', ann_data_path)
            write_pkl_safe(img_data_path, img_data)
            print('Wrote data to file: ', img_data_path)
        else:
            ann_data = read_pkl(ann_data_path)
            print('Read data from file:', ann_data_path)
            img_data = read_pkl(img_data_path)
            print('Read data from file:', img_data_path)

        assert len(self.imgs_ids) == len(ann_data), 'Invalid amount of ANN entries'
        assert len(self.imgs_ids) == len(img_data), 'Invalid amount of IMG entries'

        self.ann_data = ann_data
        self.img_data = img_data
        return

    def get_amount_difficult_anns(self):
        print('----- Start -----')
        counter_img = 0
        counter_all = 0
        counter_empty = 0

        for i, img_id in enumerate(self.imgs_ids):
            bboxes: ndarray
            cats_codes: ndarray
            cats_names: ndarray
            difficults: ndarray
            size: ndarray

            bboxes, cats_codes, cats_names, difficults, size = self.ann_data[i]
            v = np.count_nonzero(difficults)
            if v > 0:
                counter_all += v
                counter_img += 1
            if v == len(difficults):
                counter_empty += 1
            if counter_all == 1 or self.verbose:
                print('Total difficult for IMG_ID', img_id)
                print(np.column_stack((bboxes, cats_names, difficults)))

        print('----- Results -----')
        print('Total amount of difficult annotations', counter_all)
        print('Total amount of images with at least one', counter_img)
        print('Total amount of images with all of them', counter_empty)
        print('----- Finish -----')

    def get_info(self, img_id, cats_names_ann, bboxes_ann, cats_names_img, bboxes_img):
        print(f'IMG_ID: {img_id} ANN BBoxes:{len(bboxes_ann)} IMG BBoxes:{len(bboxes_img)}')
        print(f'BBoxes ANN')
        print(np.column_stack((cats_names_ann, bboxes_ann)))
        print(f'BBoxes IMG')
        print(np.column_stack((cats_names_img, bboxes_img)))
        return

    def get_amount_non_equal_bboxes_amount(self, excluded: Optional[list] = None) -> List[str]:
        print('----- Start -----')
        if excluded is None:
            excluded = []

        counter = 0
        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[i]
            bboxes_img, cats_codes_img, cats_names_img, obj_colors_img = self.img_data[i]

            if len(bboxes_ann) != len(bboxes_img):
                counter += 1
                excluded.append(img_id)
                if counter == 1 or self.verbose:
                    print('Non-equal BBoxes')
                    self.get_info(img_id, cats_names_ann, bboxes_ann, cats_names_img, bboxes_img)

        print('----- Results -----')
        print('Total amount of images with NON-EQUAL BBoxes amount', counter)
        print('----- Finish -----')
        return excluded

    def get_amount_cats_set_mismatch(self, excluded: Optional[list] = None) -> List[str]:
        print('----- Start -----')
        if excluded is None:
            excluded = []

        counter = 0
        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[i]
            bboxes_img, cats_codes_img, cats_names_img, obj_colors_img = self.img_data[i]

            unique_ann: list = list(np.unique(cats_codes_ann))
            unique_img: list = list(np.unique(cats_codes_img))

            # There were only 2 such images in trainval
            if unique_ann != unique_img:
                counter += 1
                excluded.append(img_id)
                if counter == 1 or self.verbose:
                    print('Mismatch of cats SET')
                    self.get_info(img_id, cats_names_ann, bboxes_ann, cats_names_img, bboxes_img)

        print('----- Results -----')
        print('Total amount of images with cats SET mismatch', counter)
        print('----- Finish -----')
        return excluded

    def get_amount_cats_list_mismatch(self, excluded: Optional[list] = None) -> List[str]:
        print('----- Start -----')
        if excluded is None:
            excluded = []

        counter = 0
        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            bboxes_ann, cats_codes_ann, cats_names_ann = self.ann_data[i][:3]
            bboxes_img, cats_codes_img, cats_names_img = self.img_data[i][:3]

            cats_codes_ann: list = sorted(cats_codes_ann)
            cats_codes_img: list = sorted(cats_codes_img)

            if cats_codes_ann != cats_codes_img:
                counter += 1
                excluded.append(img_id)
                if counter == 1 or self.verbose:
                    print('Mismatch of cats LIST')
                    self.get_info(img_id, cats_names_ann, bboxes_ann, cats_names_img, bboxes_img)

        print('----- Results -----')
        print('Total amount of images with cats LIST mismatch', counter)
        print('----- Finish -----')
        return excluded

    def get_amount_cat_iou_mismatch(self, excluded: Optional[list] = None) -> List[str]:
        print('----- Start -----')
        if excluded is None:
            excluded = []

        counter_anns_iou_mismatch = 0
        counter_imgs_iou_mismatch = 0

        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[i]
            bboxes_img, cats_codes_img, cats_names_img, obj_colors_img = self.img_data[i]

            t_bboxes_ann = torch.from_numpy(bboxes_ann)
            t_bboxes_img = torch.from_numpy(bboxes_img)
            ious = ops.box_iou(t_bboxes_ann, t_bboxes_img).numpy()
            ious = np.around(ious, decimals=3)
            del t_bboxes_ann, t_bboxes_img

            assert set(cats_codes_ann) == set(cats_codes_img)
            assert sorted(cats_codes_ann) == sorted(cats_codes_img)

            counter_anns_iou_mismatch_cur = 0
            cats = np.unique(cats_codes_ann)
            for c in cats:
                indexes_ann = np.where(cats_codes_ann == c)[0]
                indexes_img = np.where(cats_codes_img == c)[0]
                ious_cur = ious[indexes_ann, :][:, indexes_img]

                # x = np.arange(len(indexes_ann))
                y = np.argmax(ious_cur, axis=1)

                # Check for double choice
                # There are about 20 images with this mismatch problem
                if len(set(y)) < len(y):
                    counter_anns_iou_mismatch_cur += 1
                    if counter_anns_iou_mismatch == 0 and \
                            counter_anns_iou_mismatch_cur == 1 and \
                            self.verbose:
                        print(f'IOU mismatch of cats LIST for IMG_ID: {img_id} Category {c}')
                        # Indexes selection may be performed
                        self.draw_img_data_entry(img_id,
                                                 cats_names_ann[indexes_ann],
                                                 bboxes_ann[indexes_ann],
                                                 cats_names_img[indexes_img],
                                                 bboxes_img[indexes_img],
                                                 reason='IOU Mismatch')

                        # <<Save>> some images
                        print('IOUs')
                        print(ious_cur)
                        y_new = np.argmax(ious_cur, axis=0)
                        x_new = np.arange(len(indexes_img))
                        ious_cur_new = np.zeros_like(ious_cur)
                        ious_cur_new[y_new, x_new] = ious_cur[y_new, x_new]
                        ious_cur = ious_cur_new
                        del x_new, y_new, ious_cur_new
                        print('IOUs New')
                        print(ious_cur)

                        print('ARGMAX values OLD:', y)
                        y = np.argmax(ious_cur, axis=1)
                        print('ARGMAX values NEW:', y)
                        self.draw_img_data_entry(img_id,
                                                 cats_names_ann[indexes_ann],
                                                 bboxes_ann[indexes_ann],
                                                 cats_names_img[indexes_img],
                                                 bboxes_img[indexes_img],
                                                 reason='IOU Mismatch Two')

                        # Some images could be not <<saved>>
                        if len(set(y)) < len(y):
                            print('Improved NOT')
                        else:
                            print('Improved OK')
                        print('Finish')

            if counter_anns_iou_mismatch_cur > 0:
                counter_anns_iou_mismatch += counter_anns_iou_mismatch_cur
                counter_imgs_iou_mismatch += 1
                excluded.append(img_id)
                del counter_anns_iou_mismatch_cur

        print('----- Results -----')
        print('IOU mismatch')
        print('Total amount of anns', counter_anns_iou_mismatch)
        print('Total amount of imgs', counter_imgs_iou_mismatch)
        print('----- Finish -----')
        return excluded

    def get_amount_cat_low_iou(self, excluded: Optional[list] = None) -> List[str]:
        print('----- Start -----')
        if excluded is None:
            excluded = []

        counter_anns_low_iou = 0
        counter_imgs_low_iou_any = 0
        counter_imgs_low_iou_all = 0

        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[i]
            bboxes_img, cats_codes_img, cats_names_img, obj_colors_img = self.img_data[i]

            t_bboxes_ann = torch.from_numpy(bboxes_ann)
            t_bboxes_img = torch.from_numpy(bboxes_img)
            ious = ops.box_iou(t_bboxes_ann, t_bboxes_img).numpy()
            ious = np.around(ious, decimals=3)
            del t_bboxes_ann, t_bboxes_img

            assert set(cats_codes_ann) == set(cats_codes_img)
            assert sorted(cats_codes_ann) == sorted(cats_codes_img)

            counter_anns_low_iou_cur = 0
            cats = np.unique(cats_codes_ann)
            for c in cats:
                indexes_ann = np.where(cats_codes_ann == c)[0]
                indexes_img = np.where(cats_codes_img == c)[0]
                ious_cur = ious[indexes_ann, :][:, indexes_img]

                x = np.arange(len(indexes_ann))
                y = np.argmax(ious_cur, axis=1)

                # Check for small IOUS.
                # Possible variant to choose these small points on an instance segmentation map.
                ious_max = ious_cur[x, y]
                indexes = np.nonzero(ious_max < self.thres_iou)[0]
                amount = len(indexes)
                counter_anns_low_iou_cur += amount

                if amount > 0 and self.verbose:
                    print(f'LOW IOU for IMG_ID: {img_id} Category {c}')
                    print('IOUs')
                    print(ious_cur)
                    print('IOUs MAX')
                    print(ious_max)
                    print('Total lower than', self.thres_iou, 'amount', amount)
                    self.draw_img_data_entry(img_id,
                                             cats_names_ann[indexes_ann][x[indexes]],
                                             bboxes_ann[indexes_ann][x[indexes]],
                                             cats_names_img[indexes_img][y[indexes]],
                                             bboxes_img[indexes_img][y[indexes]],
                                             reason=f'Low IOU {self.thres_iou:.2f}')
                    print('Finish Low IOU')

            # Analysis for a single image
            if counter_anns_low_iou_cur > 0:
                excluded.append(img_id)
                counter_anns_low_iou += counter_anns_low_iou_cur

                if counter_anns_low_iou_cur == len(bboxes_ann):
                    counter_imgs_low_iou_all += 1
                else:
                    counter_imgs_low_iou_any += 1
            # End

        print('----- Results -----')
        print('Low IOU')
        print('Total amount of anns', counter_anns_low_iou)
        print('Total amount of imgs with any', counter_imgs_low_iou_any)
        print('Total amount of imgs with all', counter_imgs_low_iou_all)
        print('----- Finish -----')
        return excluded

    def get_amount_anns_imgs_min_size_ratio(self, excluded: Optional[list] = None) -> List[str]:
        print('----- Start -----')
        if excluded is None:
            excluded = []

        counter_anns = 0
        counter_imgs_any = 0
        counter_imgs_all = 0

        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            img_file_fp: str = os.path.join(self.imgs_dir_fp, img_id + '.jpg')
            w, h = imagesize.get(img_file_fp)
            img_area = w * h

            bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[i]
            t_bboxes_ann = torch.from_numpy(bboxes_ann)
            areas = ops.box_area(t_bboxes_ann).numpy()
            ratios = np.around(areas / img_area, decimals=4)
            indexes = np.where(ratios <= self.min_size_ratio)[0]
            amount = len(indexes)

            if amount > 0 and self.verbose:
                print(f'Small area of annotations for IMG_ID: {img_id}')
                print('All ratios')
                print(ratios)
                print('Total lower than', self.min_size_ratio, 'amount', amount)
                self.draw_img_data_entry(img_id,
                                         cats_names_ann[indexes],
                                         bboxes_ann[indexes],
                                         [],
                                         [],
                                         reason=f'Small size {self.min_size_ratio:.4f}')
                print('Finish Low IOU')

            counter_anns += amount
            if amount > 0:
                if self.filter_min_size_ratio:
                    excluded.append(img_id)

                if amount == len(bboxes_ann):
                    counter_imgs_all += 1
                else:
                    counter_imgs_any += 1

        print('----- Results -----')
        print(f'Small objects, min size ratio {self.min_size_ratio}')
        print(f'Excluding objects flag: {self.filter_min_size_ratio}')
        print('Total amount of anns', counter_anns)
        print('Total amount of imgs with any', counter_imgs_any)
        print('Total amount of imgs with all', counter_imgs_all)
        print('----- Finish -----')
        return excluded

    def get_amount_outside_coords(self, excluded: Optional[list] = None) -> List[str]:
        # Another method for analysis
        # Method was not used for parsing
        print('----- Start -----')

        counter_anns = 0
        counter_imgs_any = 0
        counter_imgs_all = 0

        for i, img_id in enumerate(self.imgs_ids):
            if img_id in excluded:
                if self.verbose:
                    print('Not processing IMG_ID', img_id)
                continue

            bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[i]
            bboxes_img, cats_codes_img, cats_names_img, obj_colors_img = self.img_data[i]

            # ** Match **
            bboxes1 = bboxes_ann
            bboxes2 = bboxes_img
            if len(bboxes1) != bboxes2:
                assert False

            a = bboxes1[:, 0] <= bboxes2[:, 0]
            b = bboxes1[:, 1] <= bboxes2[:, 1]
            c = bboxes1[:, 2] >= bboxes2[:, 2]
            d = bboxes1[:, 3] >= bboxes2[:, 3]

            product = a * b * c * d
            amount = np.sum(product)

            if amount > 0:
                counter_anns += amount
                # Not excluding!

                if amount == len(bboxes1):
                    counter_imgs_all += 1
                else:
                    counter_imgs_any += 1

        print('----- Results -----')
        print(f'Outside coordinates')
        print('Total amount of anns', counter_anns)
        print('Total amount of imgs with any', counter_imgs_any)
        print('Total amount of imgs with all', counter_imgs_all)
        print('----- Finish -----')
        return excluded

    def get_excluded(self) -> List[str]:
        name = str.upper(self.img_set)
        exc_data_path = os.path.join(self.root, f'EXCLUDED_{name}.json')
        del name

        if not check_file_if_exists(exc_data_path):
            # To understand their amount
            self.get_amount_difficult_anns()

            exc: List[str] = []
            exc = self.get_amount_non_equal_bboxes_amount()
            exc = self.get_amount_cats_set_mismatch(exc)
            exc = self.get_amount_cats_list_mismatch(exc)
            exc = self.get_amount_cat_iou_mismatch(exc)
            exc = self.get_amount_cat_low_iou(exc)
            exc = self.get_amount_anns_imgs_min_size_ratio(exc)

            print('Total excluded amount', len(exc))
            percent = np.around(len(exc) / len(self.imgs_ids), decimals=3)
            print('Total excluded percent', percent)
            print('IDS:', exc)

            write_json_safe(exc_data_path, exc)
            print('Wrote excluded images list to', exc_data_path)
        else:
            exc = read_json(exc_data_path)
            print('Read excluded images list from', exc_data_path)

        return exc

    def get_result_entry(self, idx) -> Union[Tuple[str, List, List, List], int]:
        img_id = self.imgs_ids[idx]

        if img_id in self.excluded:
            if self.verbose:
                print('Not processing IMG_ID', img_id)
            return 0

        this_img_sp = img_id + '.jpg'
        this_bboxes = []
        this_cat_ids = []
        this_info_isegmaps = []

        img_file_fp: str = os.path.join(self.imgs_dir_fp, img_id + '.jpg')
        w, h = imagesize.get(img_file_fp)
        img_area = w * h

        obj_isegmap_file_fp: str = os.path.join(self.obj_isegmaps_dir_fp, img_id + '.png')
        obj_isegmap = cv2.imread(obj_isegmap_file_fp, cv2.IMREAD_COLOR)[..., ::-1]

        bboxes_ann, cats_codes_ann, cats_names_ann, difficults, size = self.ann_data[idx]
        bboxes_img, cats_codes_img, cats_names_img, obj_colors_img = self.img_data[idx]

        t_bboxes_ann = torch.from_numpy(bboxes_ann)
        t_bboxes_img = torch.from_numpy(bboxes_img)
        ious = ops.box_iou(t_bboxes_ann, t_bboxes_img).numpy()
        ious = np.around(ious, decimals=3)
        areas = ops.box_area(t_bboxes_ann).numpy()
        ratios = areas / img_area
        del t_bboxes_ann, t_bboxes_img

        assert set(cats_codes_ann) == set(cats_codes_img)
        assert sorted(cats_codes_ann) == sorted(cats_codes_img)

        cats_codes_ann_unique = np.unique(cats_codes_ann)

        for cat_code_ann in cats_codes_ann_unique:
            indexes_ann = np.where(cats_codes_ann == cat_code_ann)[0]
            if self.filter_min_size_ratio:
                internal = np.where(ratios[indexes_ann] > self.min_size_ratio)[0]
                if len(internal) == 0:
                    continue
                else:
                    indexes_ann = indexes_ann[internal]

            # Cat ID
            this_cat_ids.append([self.voc_codes_coco_cats_ids[cat_code_ann]]
                                * len(indexes_ann))

            # BBoxes
            this_bboxes.append(bboxes_ann[indexes_ann])

            # Segmentation
            indexes_img = np.where(cats_codes_img == cat_code_ann)[0]
            ious_cur = ious[indexes_ann, :][:, indexes_img]

            # x = np.arange(len(indexes_ann))
            y = np.argmax(ious_cur, axis=1)
            if len(set(y)) != len(y):
                assert False

            colors = obj_colors_img[indexes_img][y]
            assert colors.ndim == 2 and colors.shape[-1] == 3
            for color in colors:
                obj_isegmap_cur = np.min(obj_isegmap == color, axis=-1)
                if obj_isegmap_cur.sum() == 0:
                    assert False, 'Empty mask'

                assert obj_isegmap_cur.shape == (h, w), 'Invalid shape after comparison'
                rle = mask.encode(np.asfortranarray(obj_isegmap_cur))
                this_info_isegmaps.append(rle)
                del obj_isegmap_cur

        if len(this_cat_ids) > 0 and len(this_bboxes) > 0:
            this_cat_ids = np.concatenate(this_cat_ids, axis=0).astype(np.int32).reshape(-1)
            this_bboxes = np.concatenate(this_bboxes, axis=0).astype(np.int32).reshape(-1, 4)

        if not (len(this_cat_ids) == len(this_bboxes) == len(this_info_isegmaps)):
            assert False

        return this_img_sp, this_bboxes, this_cat_ids, this_info_isegmaps

    def get_results(self, excluded: Optional[list] = None):
        print('----- Start -----')

        name = str.upper(self.img_set)
        parsed_data_path = os.path.join(self.root, f'PARSED_DATA_{name}.pkl')
        del name
        ignored = -1
        empty = -1

        if not os.path.exists(parsed_data_path):
            if excluded is None:
                excluded = []
            self.excluded = excluded

            # Load COCO cats
            file_sp = 'COCOCats.json'
            file_fp = os.path.join(self.root, file_sp)
            coco_cats_ids_labels = read_json(file_fp)
            coco_labels_cats_ids = {}

            for cat_id in coco_cats_ids_labels:
                name = coco_cats_ids_labels[cat_id]['name']
                name = name.replace(' ', '')
                coco_labels_cats_ids[name] = int(cat_id)
            del coco_cats_ids_labels

            extra = {'aeroplane': 'airplane',
                     'sofa': 'couch',
                     'tvmonitor': 'tv',
                     'motorbike': 'motorcycle'}

            self.voc_labels_coco_cats_ids = {}
            self.voc_codes_coco_cats_ids = {}
            for label in Colors.voc_labels:
                coco_label = label
                if label in extra:
                    coco_label = extra[label]

                assert coco_label in coco_labels_cats_ids
                self.voc_labels_coco_cats_ids[label] = coco_labels_cats_ids[coco_label]
                code = Colors.voc_labels_codes[label]
                assert isinstance(code, int)
                self.voc_codes_coco_cats_ids[code] = coco_labels_cats_ids[coco_label]

            dl = DataLoader(self, batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn_new, persistent_workers=persistent_workers)

            img_sps = []
            bboxes = []
            cat_ids = []
            info_isegmaps = []

            for i, batch in tqdm(enumerate(dl), desc='Loop over DataLoader'):
                data = batch[0]
                if isinstance(data, int) and data == 0:
                    ignored += 1
                    if self.verbose:
                        print(f'IMG {self.imgs_ids[i]} ignored')
                    continue

                this_bboxes = data[1]
                if len(this_bboxes) == 0:
                    empty += 1
                    if self.verbose:
                        print(f'IMG {self.imgs_ids[i]} empty BBoxes set')
                    continue

                this_img_sp = data[0]
                this_cat_ids = data[2]
                this_isegmaps_info = data[3]

                img_sps.append(this_img_sp)
                bboxes.append(this_bboxes)
                cat_ids.append(this_cat_ids)
                info_isegmaps.append(this_isegmaps_info)

            data = [img_sps, bboxes, cat_ids, info_isegmaps]
            write_pkl_safe(parsed_data_path, data)
            print('Wrote data to', parsed_data_path)
        else:
            data = read_pkl(parsed_data_path)
            img_sps, bboxes, cat_ids, info_isegmaps = data
            print('Read data from', parsed_data_path)

        print('----- Results -----')
        print('Gathered all results')
        print(f'Total Ignored {ignored:4} images')
        print(f'Total Empty   {empty:4} images')
        print(f'Total Normal  {len(bboxes)} images')
        print('----- Finish -----')
        return img_sps, bboxes, cat_ids, info_isegmaps

    def draw_img_data_entry(self, img_id, cats_ann, bboxes_ann, cats_img, bboxes_img, reason: str, draw_all=False):
        save_dir = os.path.join(self.root, reason)
        create_empty_dir_safe(save_dir)

        img_file_fp: str = os.path.join(self.imgs_dir_fp, img_id + '.jpg')
        img = cv2.imread(img_file_fp, cv2.IMREAD_COLOR)[..., ::-1]

        bboxes_ann_ia = self.get_bboxes_on_img_from_yxyx(img, bboxes_ann, cats_ann, 'ANN')
        bboxes_img_ia = self.get_bboxes_on_img_from_yxyx(img, bboxes_img, cats_img, 'IMG')

        img = self.draw_on_img(img, bboxes_ann_ia, color='g')
        img = self.draw_on_img(img, bboxes_img_ia, color='r')

        self.save(img, save_dir=save_dir, img_name=f'{img_id}_IMG_BBoxes.png')

        if draw_all:
            obj_isegmap_file_fp: str = os.path.join(self.obj_isegmaps_dir_fp, img_id + '.png')
            cls_isegmap_file_fp: str = os.path.join(self.cls_isegmaps_dir_fp, img_id + '.png')

            obj_isegmap = cv2.imread(obj_isegmap_file_fp, cv2.IMREAD_COLOR)[..., ::-1]
            cls_isegmap = cv2.imread(cls_isegmap_file_fp, cv2.IMREAD_COLOR)[..., ::-1]

            obj_isegmap = self.draw_on_img(obj_isegmap, bboxes_ann_ia, color='g')
            obj_isegmap = self.draw_on_img(obj_isegmap, bboxes_img_ia, color='r')

            cls_isegmap = self.draw_on_img(cls_isegmap, bboxes_ann_ia, color='g')
            cls_isegmap = self.draw_on_img(cls_isegmap, bboxes_img_ia, color='r')

            self.save(obj_isegmap, save_dir=save_dir, img_name=f'{img_id}_OBJ_BBoxes.png')
            self.save(cls_isegmap, save_dir=save_dir, img_name=f'{img_id}_CLS_BBoxes.png')

        return

    def copy_excluded(self, excluded: List[str]):
        assert isinstance(excluded, list)
        if len(excluded) == 0:
            print('List for excluded images is empty!')

        save_dir = os.path.join(self.root, 'Excluded')
        create_empty_dir_safe(save_dir)

        for img_id in excluded:
            ann_file_fp: str = os.path.join(self.ann_dir_fp, img_id + '.xml')
            ann_file_fp_new: str = os.path.join(save_dir, img_id + 'XML.xml')

            img_file_fp: str = os.path.join(self.imgs_dir_fp, img_id + '.jpg')
            img_file_fp_new: str = os.path.join(save_dir, img_id + '.RGB.Excluded.jpg')

            obj_isegmap_file_fp: str = os.path.join(self.obj_isegmaps_dir_fp, img_id + '.png')
            obj_isegmap_file_fp_new: str = os.path.join(save_dir, img_id + '.OBJ.Excluded.jpg')

            cls_isegmap_file_fp: str = os.path.join(self.cls_isegmaps_dir_fp, img_id + '.png')
            cls_isegmap_file_fp_new: str = os.path.join(save_dir, img_id + '.CLS.Excluded.jpg')

            if not os.path.isfile(ann_file_fp_new):
                shutil.copyfile(ann_file_fp, ann_file_fp_new)
            if not os.path.isfile(img_file_fp_new):
                shutil.copyfile(img_file_fp, img_file_fp_new)
            if not os.path.isfile(obj_isegmap_file_fp_new):
                shutil.copyfile(obj_isegmap_file_fp, obj_isegmap_file_fp_new)
            if not os.path.isfile(cls_isegmap_file_fp_new):
                shutil.copyfile(cls_isegmap_file_fp, cls_isegmap_file_fp_new)

        return

    @staticmethod
    def get_bboxes_on_img_from_yxyx(img: ndarray,
                                    bboxes: Union[ndarray, List[ndarray]],
                                    cat_ids: Optional[Union[int, str, List[int], List[str], ndarray]] = None,
                                    cat_id_comment: str = '') -> BoundingBoxesOnImage:
        if isinstance(bboxes, BoundingBoxesOnImage):
            return bboxes

        # Make it safe
        bboxes = np.reshape(bboxes, newshape=(-1, 4))
        if cat_ids is None:
            cat_ids = ['UNK' for _ in range(len(bboxes))]
        cat_ids = np.array(cat_ids, dtype=str).reshape(-1).tolist()
        bounding_boxes = []
        for i in range(len(bboxes)):
            label = cat_ids[i]
            if len(cat_id_comment) > 0:
                label = cat_ids[i] + ' ' + cat_id_comment
            bounding_boxes.append(BoundingBox(y1=bboxes[i][0],
                                              x1=bboxes[i][1],
                                              y2=bboxes[i][2],
                                              x2=bboxes[i][3],
                                              label=label))
        # Not converting in a batch mode for a better
        bounding_boxes_on_image = BoundingBoxesOnImage(bounding_boxes, shape=img.shape)
        return bounding_boxes_on_image

    @staticmethod
    def draw_on_img(img: ndarray,
                    bboxes: Optional[Union[List[ndarray], ndarray, BoundingBoxesOnImage]] = None,
                    cat_ids: Optional[Union[List[Union[int, str, ndarray]], ndarray]] = None,
                    cat_id_comment: str = '',
                    color='g') -> ndarray:
        # Other imgaug visualization params may be set in future with method signature
        if isinstance(color, str):
            assert len(color) == 1, f'Invalid color {len(color)}'
            if color == 'g':
                color = (0, 255, 0)
            elif color == 'r':
                color = (255, 0, 0)
            else:
                color = (0, 0, 0)

        if bboxes is not None and len(bboxes) > 0:
            bbs = VOCDSParse.get_bboxes_on_img_from_yxyx(img, bboxes, cat_ids, cat_id_comment)
            img = bbs.draw_on_image(img, color=color)

        return img

    @staticmethod
    def save(img, save_dir, img_name):
        assert save_dir is not None
        assert img_name is not None
        img_name = os.path.join(save_dir, img_name)
        plt.imsave(img_name, img)
        plt.clf()

        return
