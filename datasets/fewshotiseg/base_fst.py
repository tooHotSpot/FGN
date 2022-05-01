import os
import gc
import contextlib
import warnings
from tqdm import tqdm
from printy import printy

# For faster and better types annotation
from typing import List, Tuple, Dict, Union, Optional
from numpy import ndarray
from torch import Tensor

# Basic ML modules
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import imagesize

# DL frameworks
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
from imgaug import BoundingBox, BoundingBoxesOnImage
from imgaug import SegmentationMapsOnImage

# Own packages
from cp_utils.cp_time import time_log_fancy
from cp_utils.cp_dir_file_ops import create_empty_dir_unsafe, create_empty_dir_safe
from cp_utils.cp_dir_file_ops import check_file_if_exists
from cp_utils.cp_dir_file_ops import read_pkl, write_pkl_safe
from cp_utils.create_img_from_chars import get_new_shape

from datasets.fewshotiseg.fsisegeval import FSISEGEval
from pycocotools.mask import decode


class BaseFewShotISEG(Dataset):
    # region Params to set in a main (child) class
    root: str
    target_size: int
    max_size: int
    spp_img_size: int

    mean: ndarray
    std: ndarray

    augs_qry: iaa.Sequential
    augs_spp: iaa.Sequential

    cats_total_amount: int
    cat_names_to_ids: Dict[str, int]
    cat_ids_to_names: Dict[int, str]
    cats_novel: ndarray

    imgs_dir_fp: str
    # merged_ds_imgs_dir_fp: Optional[str] = None
    imgs_dir_fps: Optional[list] = None
    imgs_sps = []
    bboxes = []
    cat_ids = []
    info_isegmaps = []
    transforms: Optional[transforms.Compose]
    # endregion

    # region Default BaseFewShotISEG params
    sub_sample_ratio = 16
    n_ways = 3
    k_shots = 3
    # Resize and Pad ops for support images to collect several in a batch for a high inference speed.
    resize_visualize = iaa.Resize({'longer-side': 256, 'shorter-side': 'keep-aspect-ratio'})
    pad_visualize = iaa.CenterPadToFixedSize(width=256, height=256, pad_mode='constant')
    verbose = False
    # endregion

    # region Params computed during initialization and data loading
    setup: str
    cats_base__total: int
    cats_novel_total: int
    cats_base_: ndarray
    # Vector with 0 on base classes positions and 1 on novel class positions
    cat_is_base_: ndarray
    cats_to_save: ndarray
    cats_to_del_: ndarray
    cats_to_save_bool: ndarray

    qrys_parents_: List[Dict[str, Union[Dict, str]]]
    qrys_children: List[List[int]]
    order: ndarray
    order_initial: ndarray
    # To get instances annotations immediately
    insts: List[Dict[str, Union[int, list, str, ndarray]]]
    # To sample support set
    cats_insts_list: List[List[int]]
    # Ops for support images sampling
    resize_spp: iaa.Resize
    pad_spp: Union[iaa.Pad, iaa.CenterPadToFixedSize]
    # Data bag
    databag_fp: str
    # endregion

    # region Set up with config (required for categories sampling)
    ds_base_: str
    ds_base__subset: str
    ds_novel: str
    ds_novel_subset: str

    sampling_origin_ds: str
    sampling_origin_ds_subset: str
    sampling_cats: str
    sampling_cats_options = ('base_', 'novel', 'all')

    first_parents__only = 0
    first_children_only = 0
    sampling_scenario = 'parents'
    sampling_scenario_options = ('parents', 'children')
    repeats = 1
    shuffle = False

    augment_qry = False
    augment_spp = False
    # Overfit mode may be created with other settings set appropriately
    overfit_sample_mode = False
    overfit_sample = None

    qry_cats_choice_remove = False
    qry_cats_choice_random = False
    qry_cats_order_shuffle = True
    delete_qry_insts_in_spp_insts_on_train = True
    spp_random = True
    get_plot = False
    finetune = 'Ignore'
    finetune_options = ('Ignore', 'Select', 'Use')
    upper_ds: 'BaseFewShotISEG' = None
    merged_ds: 'BaseFewShotISEG' = None
    # 1 / (1 + 2 * OFFSET) = spp_fill_ratio
    spp_fill_ratio = 0.8
    spp_crop_square = True
    offset_ratio = 0
    suffix = 'SuffixUnset'
    batch = 1
    # Of self.order length
    ar_group_idx = []
    # Target and max sizes for images in this group
    ar_group_new_hws: ndarray
    ann_min_size_ratio: float = 0.005

    # endregion

    def denormalize(self, img: Union[ndarray, Tensor], **kwargs) -> Union[ndarray, Tensor]:
        raise NotImplementedError

    def get_isegmap(self, img: ndarray, bbox: Union[list, ndarray], info: Union[list, ndarray]) -> ndarray:
        raise NotImplementedError

    def v_print(self, msg, color='y'):
        if self.verbose:
            printy(msg, f'{color}BU')

    @staticmethod
    def a_print(msg, color='n'):
        # Always print
        printy(msg, f'{color}BU')

    @staticmethod
    def e_print(msg, color='r'):
        # Error to print
        printy(msg, f'{color}BU')

    def __init__(self, config: dict):
        super(BaseFewShotISEG, self).__init__()
        self.v_print('Initializing keys')

        # Compare and change only those values which are defined for the dataset class
        self.verbose = config.get('verbose', self.verbose)
        for key in config:
            value = config[key]
            try:
                req = type(getattr(self, key))
                real = type(value)
                assert isinstance(value, req), f'Types for the {key} do not match, ' \
                                               f'type {type(config[key])} required instead of {real}'
            except AttributeError:
                self.v_print(f'Could not get variable type for var {key}')
                pass
            setattr(self, key, value)
            self.v_print(f'Setting attr {key: <50} => {getattr(self, key)}')

        # region Checks
        # The dataset_novel and dataset_base_ define the base/novel class selection criteria
        # Subset knowledge is required to select data to merge both datasets
        if self.ds_base_ == 'COCO':
            assert self.ds_novel in ('COCO', 'VOC')
            assert self.ds_base__subset in ('train', 'val')
        elif self.ds_base_ == 'VOC':
            assert self.ds_novel in ('COCO', 'VOC')
            assert self.ds_base__subset in ('train', 'val', 'trainval')
        elif self.ds_base_ in ('OMNIISEG', 'MNISTISEG'):
            assert self.ds_novel in ('OMNIISEG', 'MNISTISEG')
            assert self.ds_base__subset in ('train', 'val', 'test')

        if self.ds_novel == 'COCO':
            assert self.ds_novel_subset in ('train', 'val')
        elif self.ds_novel == 'VOC':
            assert self.ds_novel_subset in ('train', 'val', 'trainval')
        elif self.ds_novel in ('OMNIISEG', 'MNISTISEG'):
            assert self.ds_novel_subset in ('train', 'val', 'test')

        self.setup = self.ds_base_ + '2' + self.ds_novel
        assert self.setup in ('COCO2VOC', 'VOC2VOC',
                              'OMNIISEG2OMNIISEG', 'OMNIISEG2MNISTISEG',
                              'MNISTISEG2MNISTISEG', 'MNISTISEG2OMNIISEG')

        # COCO and VOC test sets are absent
        # for synthetic datasets I have not sampled a test subset
        if self.sampling_origin_ds == 'COCO':
            assert self.sampling_origin_ds_subset in ('train', 'val')
        elif self.sampling_origin_ds == 'VOC':
            assert self.sampling_origin_ds_subset in ('train', 'val', 'trainval')
        elif self.sampling_origin_ds in ('OMNIISEG', 'MNISTISEG'):
            assert self.sampling_origin_ds_subset in ('train', 'val', 'test')
        assert self.sampling_cats in self.sampling_cats_options, \
            f'Invalid value: {self.sampling_cats}'

        # DatasetA2DatasetA - no other could be
        # ('VOC2VOC', 'COCO', 'base')
        # ('VOC2VOC', 'COCO', 'novel')
        if self.ds_base_ == self.ds_novel:
            assert self.ds_base_ == self.sampling_origin_ds
        # DatasetA2DatasetB - depends on the stage
        # Some combinations are checked earlier
        if self.ds_base_ != self.ds_novel:
            assert self.sampling_origin_ds in (self.ds_base_, self.ds_novel)

        # Integration checks
        assert (self.setup, self.sampling_origin_ds, self.sampling_cats) not in [
            # There are no instances or base classes in COCO2VOC setting in VOC and similarly
            ('COCO2VOC', 'VOC', 'base_'),
            ('OMNIISEG2MNISTISEG', 'MNISTISEG', 'base_'),
            ('OMNIISEG2MNISTISEG', 'OMNIISEG', 'novel'),
            ('MNISTISEG2OMNIISEG', 'OMNIISEG', 'base_'),
            ('MNISTISEG2OMNIISEG', 'MNISTISEG', 'novel'),
        ]

        assert self.finetune in self.finetune_options, f'Invalid option {self.finetune}'
        # endregion

        if not os.path.exists(self.root):
            create_empty_dir_safe(self.root)
        self.suffix = f'{self.sampling_origin_ds}_' \
                      f'{self.sampling_origin_ds_subset}_' \
                      f'{self.sampling_cats}_' \
                      f'FilterArea{self.ann_min_size_ratio:0.03f}_' \
                      f'FT_{self.finetune}'

        if self.finetune != 'Ignore':
            self.suffix += f'_K{self.k_shots}'
        file_sp = f'{self.setup}_{self.suffix}.pkl'

        self.databag_fp = os.path.join(self.root, file_sp)

        assert isinstance(self.spp_fill_ratio, float) and 0.5 <= self.spp_fill_ratio <= 1.0
        self.offset_ratio = np.around(1 / (2 * self.spp_fill_ratio) - 0.5, decimals=2)

    def cats_selection(self):
        if self.sampling_cats == 'all':
            self.cats_novel = np.array([], dtype=np.int32)

        self.cat_is_base_ = np.ones(self.cats_total_amount, dtype=np.bool)
        self.cat_is_base_[self.cats_novel] = 0
        self.cats_base_ = np.where(self.cat_is_base_)[0]
        self.cats_base__total = len(self.cats_base_)
        self.cats_novel_total = len(self.cats_novel)

        if self.sampling_cats == 'base_':
            self.cats_to_save, self.cats_to_del_ = self.cats_base_, self.cats_novel
        elif self.sampling_cats == 'novel':
            self.cats_to_save, self.cats_to_del_ = self.cats_novel, self.cats_base_
        elif self.sampling_cats == 'all':
            # Assume we always have the non-stopping row of the categories
            self.cats_to_save = np.arange(self.cats_total_amount).astype(np.int32)
            self.cats_to_del_ = np.array([], dtype=np.int32)
        else:
            warnings.warn('Invalid options in cases' + self.sampling_cats)
            assert False

        self.cats_to_save_bool = np.zeros(self.cats_total_amount, dtype=np.bool)
        self.cats_to_save_bool[self.cats_to_save] = True
        # Additional test
        if self.sampling_cats != 'all':
            assert len(set(self.cats_to_save) & set(self.cats_to_del_)) == 0, \
                'Groups of categories intersect but do not have to'

    def load_dataset(self):
        self.cats_selection()

        # Additional test to control the serial dataset loading
        # Both have to be initialized on previous initialization steps
        assert self.cats_to_save is not None
        assert self.cats_to_del_ is not None

        if not os.path.exists(self.databag_fp):
            indexes_to_del = np.zeros(len(self.cat_ids), dtype=np.bool)
            if self.finetune == 'Ignore':
                pass
            else:
                from datasets.fewshotiseg.fs_selection import select_indices
                selected_indices = select_indices(self)
                if self.finetune == 'Use':
                    indexes_to_del = np.zeros(len(self.cat_ids), dtype=np.bool)
                    indexes_to_del[selected_indices] = True
                    self.a_print('Using finetune images')
                elif self.finetune == 'Select':
                    indexes = np.ones(len(self.cat_ids), dtype=np.bool)
                    indexes[selected_indices] = False
                    indexes_to_del = np.sort(np.nonzero(indexes)[0])[::-1]
                    indexes_to_del = indexes_to_del.astype(np.int32)

                    for i in indexes_to_del:
                        del self.imgs_sps[i]
                        del self.bboxes[i]
                        del self.cat_ids[i]
                        del self.info_isegmaps[i]
                del selected_indices

            self.v_print(f'Collecting the databag {self.databag_fp}')
            # Iterate over the whole dataset and create the lookup table for
            # the support instances sampling. The list of lists seems to be better.
            self.qrys_parents_ = []
            self.qrys_children = []
            self.insts = []
            # List could be generated from the set
            self.cats_insts_list = [[] for _ in range(self.cats_total_amount)]

            ann_min_size_ratio_kept = 0
            ann_min_size_ratio_removed = 0
            for i in tqdm(range(len(self.imgs_sps))):
                skip = False
                add_inst = True
                if self.finetune == 'Use' and indexes_to_del[i]:
                    skip = True
                if self.finetune == 'Use' and not indexes_to_del[i]:
                    add_inst = False

                # print('Checking path', i, self.imgs_paths[i])
                num_parent_qry = len(self.qrys_parents_)
                current_obj = {
                    'img_sp': self.imgs_sps[i],
                    'cats_dict': dict(),
                    'nums_children_qrys': []
                }
                if self.imgs_dir_fps is not None:
                    current_obj['img_dir_fp'] = self.imgs_dir_fps[i]
                for j in range(len(self.cat_ids[i])):
                    inst_id = len(self.insts)
                    cat_id = int(self.cat_ids[i][j])
                    # Carries out whole work of the old delete_if_possible() method
                    if not self.cats_to_save_bool[cat_id]:
                        continue
                    imgs_dir_fp = current_obj.get('img_dir_fp', self.imgs_dir_fp)
                    img_fp = os.path.join(imgs_dir_fp, self.imgs_sps[i])
                    w, h = imagesize.get(img_fp)
                    y1, x1, y2, x2 = np.array(self.bboxes[i][j]).astype(np.float32)
                    ratio = (x2 - x1) * (y2 - y1) / (w * h)
                    if add_inst:
                        if ratio >= self.ann_min_size_ratio:
                            self.cats_insts_list[cat_id].append(inst_id)
                            ann_min_size_ratio_kept += 1
                        else:
                            ann_min_size_ratio_removed += 1

                    if cat_id not in current_obj['cats_dict']:
                        current_obj['cats_dict'][cat_id] = []
                    current_obj['cats_dict'][cat_id].append(inst_id)

                    self.insts.append({
                        'num_parent_qry': num_parent_qry,
                        'bbox': self.bboxes[i][j],
                        'cat_id': self.cat_ids[i][j],
                        'info_isegmap': self.info_isegmaps[i][j],
                    })
                    if skip:
                        self.insts[-1].pop('num_parent_qry')
                        self.insts[-1]['ft_img_sp'] = self.imgs_sps[i]
                        # self.insts[-1]['ft_img_dir_fp'] = self.imgs_dir_fps[i]
                        # TODO: select right variant
                        self.insts[-1]['ft_img_dir_fp'] = current_obj.get('img_dir_fp', self.imgs_dir_fp)

                if len(current_obj['cats_dict']) == 0:
                    continue
                if skip:
                    continue
                # Since cat_ids may contain several examples of the same class (category),
                # it is required to choose only unique
                current_obj['cats_unique_sorted'] = list(set(current_obj['cats_dict']))
                for cat_id in current_obj['cats_unique_sorted']:
                    current_obj['nums_children_qrys'].append(len(self.qrys_children))
                    self.qrys_children.append([num_parent_qry, cat_id])

                self.qrys_parents_.append(current_obj)

            self.v_print(f'Collected!')
            self.v_print(f'Amount of parents query    {len(self.qrys_parents_)}')
            self.v_print(f'Amount of children query   {len(self.qrys_children)}')
            self.v_print(f'Length of cats_insts_sets  {len(self.cats_insts_list)}')
            self.v_print(f'Amount of insts            {len(self.insts)}')
            # (COCO-train-base) Amount of insts smaller than 0.005, kept 231366 removed 135789
            # (COCO-val-base) Amount of insts smaller than 0.005, kept 9824 removed 6006

            self.a_print(f'Amount of insts smaller than {self.ann_min_size_ratio}, '
                         f'kept {ann_min_size_ratio_kept} '
                         f'removed {ann_min_size_ratio_removed}')

            data = (
                self.qrys_parents_, self.qrys_children,
                self.cats_insts_list, self.insts
            )
            write_pkl_safe(self.databag_fp, data)
            self.a_print(f'Written data to {self.databag_fp}')
            self.imgs_sps.clear()
            self.bboxes.clear()
            self.cat_ids.clear()
            self.info_isegmaps.clear()
            gc.collect()
        else:
            self.v_print(f'Loading the databag {self.databag_fp}')
            data = read_pkl(self.databag_fp)
            (self.qrys_parents_, self.qrys_children,
             self.cats_insts_list, self.insts) = data
            self.a_print(f'Loaded the databag from {self.databag_fp}')

        self.v_print('Setting the attributes of the dataset')
        self.a_print(f'Total qrys_parents_: {len(self.qrys_parents_)}')
        self.a_print(f'Total qrys_children: {len(self.qrys_children)}')

        assert self.sampling_scenario in self.sampling_scenario_options

        total = -1
        if self.sampling_scenario == 'parents':
            total = len(self.qrys_parents_)
        elif self.sampling_scenario == 'children':
            total = len(self.qrys_children)
        else:
            assert False
        self.order = np.arange(total)
        self.a_print(f'Sampling by {self.sampling_scenario}')

        initial_order_len = len(self.order)
        if 0 < self.first_parents__only <= len(self.qrys_parents_):
            if self.sampling_scenario == 'parents':
                self.order = self.order[:self.first_parents__only]
            else:
                limit = self.qrys_parents_[:self.first_parents__only][-1]['nums_children_qrys'][-1]
                self.order = self.order[:(limit + 1)]
        else:
            self.first_parents__only = len(self.qrys_parents_)

        if 0 < self.first_children_only <= len(self.qrys_children):
            if self.sampling_scenario == 'parents':
                limit = self.qrys_children[:self.first_children_only][-1][0]
                self.order = self.order[:(limit + 1)]
            else:
                self.order = self.order[:self.first_children_only]
        else:
            self.first_children_only = len(self.qrys_children)
        self.a_print(f'Order by {self.sampling_scenario}  reduced: {initial_order_len} => {len(self.order)}')

        # Select sampling scenario
        if 1 <= self.repeats <= 100:
            self.order = np.tile(self.order, reps=self.repeats)
        else:
            self.v_print(f'Invalid or strange value {self.repeats} for arg self.repeats')
            self.repeats = 1
        self.a_print(f'Repeating the set of entries {self.repeats} times')

        self.resize_spp = iaa.Resize({'longer-side': self.spp_img_size,
                                      'shorter-side': 'keep-aspect-ratio'})
        self.pad_spp = iaa.CenterPadToFixedSize(width=self.spp_img_size,
                                                height=self.spp_img_size,
                                                pad_mode='constant')
        self.order_initial = np.array(self.order)
        self.reshuffle()

    # def get_arb_img_fp(self, img_sp):
    #     img_fp = os.path.join(self.imgs_dir_fp, img_sp)
    #     if not check_file_if_exists(img_fp):
    #         if self.merged_ds_imgs_dir_fp is not None:
    #             img_fp = os.path.join(self.merged_ds_imgs_dir_fp, img_sp)
    #         if not check_file_if_exists(img_fp):
    #             assert False
    #
    #     return img_fp

    def get_arb_img_fp_new(self, idx_parent):
        img_dir_fp = self.qrys_parents_[idx_parent].get('img_dir_fp', self.imgs_dir_fp)
        img_sp = self.qrys_parents_[idx_parent]['img_sp']

        # if 'img_dir_fp' in self.qrys_parents_[idx_parent]:
        #     img_dir_fp: str = self.qrys_parents_[idx_parent]['img_dir_fp']
        # else:
        #     img_dir_fp = self.imgs_dir_fp

        img_fp = os.path.join(img_dir_fp, img_sp)
        return img_fp

    def prepare_to_merge(self):
        # noinspection PyCallingNonCallable
        inner_ds = self.inner_ds_cl(imgs_set=self.sampling_origin_ds_subset, read_data=True)
        self.imgs_sps = inner_ds.imgs_sps
        self.bboxes = inner_ds.bboxes
        self.cat_ids = inner_ds.cat_ids
        self.info_isegmaps = inner_ds.info_isegmaps

        from datasets.fewshotiseg.fs_selection import select_indices
        selected_indices = select_indices(self)
        self.a_print(f'Total selected indices {len(selected_indices)}')

        indexes = np.ones(len(self.cat_ids), dtype=np.bool)
        indexes[selected_indices] = False
        indexes_to_del = np.sort(np.nonzero(indexes)[0])[::-1]
        indexes_to_del = indexes_to_del.astype(np.int32)

        for i in indexes_to_del:
            del self.imgs_sps[i]
            del self.bboxes[i]
            del self.cat_ids[i]
            del self.info_isegmaps[i]

        print('Deleted all redundant and ready to merge')

    def merge_ds(self, ds: 'BaseFewShotISEG'):
        # Old merge style (in main.py)
        # train_ds.merged_ds = train_ds_ft
        # train_ds_ft.upper_ds = train_ds
        # train_ds.reshuffle()

        print('Merging datasets params')
        assert self.setup == ds.setup
        print('Setup', self.setup)

        assert self.finetune == ds.finetune == 'Select'
        print('FT', self.finetune)

        assert self.sampling_cats != ds.sampling_cats
        print('* Head DS', self.sampling_origin_ds, self.sampling_cats)
        print('* Tail DS', ds.sampling_origin_ds, ds.sampling_cats)

        assert self.ann_min_size_ratio == ds.ann_min_size_ratio
        print('AnnMinSizeRatio', self.ann_min_size_ratio)

        assert self.sampling_scenario == ds.sampling_scenario
        print('Sampling scenario', self.sampling_scenario)

        assert self.n_ways == ds.n_ways
        assert self.k_shots == ds.k_shots
        print('N ways', self.n_ways)
        print('K shots', self.k_shots)

        file_sp = f'{self.setup}_' \
                  f'{self.sampling_origin_ds}_' \
                  f'{self.sampling_origin_ds_subset}_' \
                  f'{self.sampling_cats}_' \
                  f'MERGED_' \
                  f'{ds.sampling_origin_ds}_' \
                  f'{ds.sampling_origin_ds_subset}_' \
                  f'{ds.sampling_cats}_' \
                  f'_SETTING_' \
                  f'All_' \
                  f'FT_Ignore_' \
                  f'K{self.k_shots}' \
                  f'.pkl'
        databag_fp = os.path.join(self.root, file_sp)

        if not os.path.exists(databag_fp):
            self.prepare_to_merge()
            ds.prepare_to_merge()

            length0 = len(self.imgs_sps)
            length1 = len(ds.imgs_sps)
            self.imgs_dir_fps = [self.imgs_dir_fp] * length0 + \
                                [ds.imgs_dir_fp] * length1
            self.imgs_sps = self.imgs_sps + ds.imgs_sps
            self.bboxes = self.bboxes + ds.bboxes
            self.cat_ids = self.cat_ids + ds.cat_ids
            self.info_isegmaps = self.info_isegmaps + ds.info_isegmaps

        self.sampling_cats = 'all'
        self.finetune = 'Ignore'
        self.databag_fp = databag_fp
        self.first_parents__only = 0
        self.first_children_only = 0
        self.verbose = True
        self.load_dataset()
        self.verbose = False
        # Remove from the memory
        self.imgs_sps = None
        self.bboxes = None
        self.cat_ids = None
        self.info_isegmaps = None
        self.imgs_dir_fps = None
        gc.collect()

    def reshuffle(self, e=8):
        """

        :param e: random seed, mimics to epoch
        :return:
        """
        if self.batch == 1 or self.sampling_origin_ds in ('OMNIISEG', 'MNISTISEG'):
            self.a_print(f'Using batch {self.batch} since {self.sampling_origin_ds}')
            self.order = self.order_initial.copy()
            if self.merged_ds is not None:
                self.merged_ds.order = self.merged_ds.order_initial.copy()
                self.order = np.arange(len(self.order_initial) + len(self.merged_ds.order_initial))

            if self.shuffle:
                order = list(self.order)
                # Just a formula
                v = (2 ** e) % 1000
                random.Random(v).shuffle(order)
                self.order = np.array(order, dtype=np.int32)
        else:
            order = self.order_initial.copy()
            self.order = self.order_initial.copy()
            if self.merged_ds is not None:
                self.merged_ds.order = self.merged_ds.order_initial.copy()
                order = np.arange(len(self.order_initial) + len(self.merged_ds.order_initial))

            ar_order_values = []
            ar_order_fpaths = []
            for idx in order:
                path = self.__getitem__(idx, path_only=True)
                width, height = imagesize.get(path)
                ar_value = width / height
                ar_order_values.append(ar_value)
                ar_order_fpaths.append(path)
            ar_order_values = np.around(ar_order_values, decimals=1)

            unique_ar_values = sorted(np.unique(ar_order_values))
            unique_ar_indexes = {unique_ar_values[i]: i for i in range(len(unique_ar_values))}

            print(f'Aspect Ratios MIN {unique_ar_values[0]} MAX {unique_ar_values[-1]}')
            # Define list of lists
            ar_groups_order_indexes: List[List[int]] = []
            ar_groups_order_fpaths: List[List[int]] = []
            ar_groups_group_indexes: List[List[int]] = []
            ar_groups_hws_new = []
            for i in range(len(unique_ar_values)):
                ar_groups_order_indexes.append([])
                ar_groups_order_fpaths.append([])
                ar_groups_group_indexes.append([])
                w = 100
                h = w / unique_ar_values[i]
                h_new, w_new = get_new_shape(h, w, self.target_size, self.max_size)
                ar_groups_hws_new.append((h_new, w_new))

            # Put into lists
            for i in range(len(ar_order_values)):
                ar_value = ar_order_values[i]
                group_index = unique_ar_indexes[ar_value]

                # NEW LINES
                idx = order[i]
                ar_groups_order_indexes[group_index].append(idx)
                ar_groups_group_indexes[group_index].append(group_index)
                ar_groups_order_fpaths[group_index].append(ar_order_fpaths[i])

            for i in range(len(unique_ar_values)):
                # i in a group index
                group_np = np.array(ar_groups_order_indexes[i])
                group_path = np.array(ar_groups_order_fpaths[i])
                group_group_indexes = np.array(ar_groups_group_indexes[i])

                total = len(group_np)
                remain = total % self.batch
                elements_all = np.arange(total)
                if remain != 0:
                    # if self.shuffle:
                    #     elements = random.choices(elements_all, k=(self.batch - remain))
                    # else:
                    #     k = (self.batch - remain)
                    #     elements = elements_all[:k]
                    elements = random.choices(elements_all, k=(self.batch - remain))
                    elements_all = np.concatenate((elements_all, elements)).reshape(-1)
                if self.shuffle:
                    random.shuffle(elements_all)
                ar_groups_order_indexes[i] = group_np[elements_all]
                ar_groups_order_fpaths[i] = group_path[elements_all]
                ar_groups_group_indexes[i] = group_group_indexes[elements_all]

            ar_groups_hws_new = np.around(np.array(ar_groups_hws_new) / self.sub_sample_ratio) * self.sub_sample_ratio
            self.ar_group_new_hws = np.array(ar_groups_hws_new).reshape(-1, 2).astype(np.int32)

            ar_groups_order_indexes_flat_list = list(itertools.chain.from_iterable(ar_groups_order_indexes))
            ar_groups_group_indexes_flat_list = list(itertools.chain.from_iterable(ar_groups_group_indexes))
            ar_groups_order_fpaths_flat_list = list(itertools.chain.from_iterable(ar_groups_order_fpaths))

            ar_groups_order_indexes_flat_np = np.array(ar_groups_order_indexes_flat_list).astype(np.int32)
            ar_groups_group_indexes_flat_np = np.array(ar_groups_group_indexes_flat_list).astype(np.int32)
            ar_groups_order_fpaths_flat_np = np.array(ar_groups_order_fpaths_flat_list).astype(np.str)

            # Convert and divide to chunks
            ar_groups_order_indexes_chunks = np.array(ar_groups_order_indexes_flat_np).reshape(-1, self.batch)
            ar_groups_group_indexes_chunks = np.array(ar_groups_group_indexes_flat_np).reshape(-1, self.batch)
            ar_groups_order_fpaths_chunks = np.array(ar_groups_order_fpaths_flat_np).reshape(-1, self.batch)

            # Check same length
            a = ar_groups_order_fpaths_chunks.__len__()
            b = ar_groups_order_fpaths_chunks.__len__()
            c = ar_groups_order_fpaths_chunks.__len__()
            assert a == b == c

            # Shuffle chunks
            indexes_chunks = np.arange(a)
            if self.shuffle:
                indexes_chunks = list(indexes_chunks)
                random.shuffle(indexes_chunks)
                indexes_chunks = np.array(indexes_chunks)

            self.ar_groups_order_indexes_all = ar_groups_order_indexes_chunks[indexes_chunks].reshape(-1)
            self.ar_groups_group_indexes_all = ar_groups_group_indexes_chunks[indexes_chunks].reshape(-1)
            self.ar_groups_order_fpaths_all = ar_groups_order_fpaths_chunks[indexes_chunks].reshape(-1)

            self.order = self.ar_groups_order_indexes_all

            self.a_print(f'Divided into {len(indexes_chunks)} chunks')
            old_amount = len(self.order_initial)
            new_amount = len(self.order)
            self.a_print(f'Finished, elements amount {old_amount} (OLD) -> {new_amount} (NEW)')
        self.a_print(f'Reshuffled the dataset with Batch Size {self.batch}')

    @staticmethod
    def augment_with_imgaug(augs_series: iaa.Sequential,
                            img: ndarray,
                            bboxes: ndarray,
                            isegmaps: ndarray) -> Tuple[ndarray, ...]:
        bad_augment = False

        bboxes_ia: BoundingBoxesOnImage = BaseFewShotISEG.get_bboxes_on_img_from_yxyx(img, bboxes=bboxes)
        isegmaps_ia: List[SegmentationMapsOnImage] = \
            BaseFewShotISEG.get_isegmaps_on_img_multiple(img, isegmaps)

        seq_det = augs_series.to_deterministic()
        img_new, bboxes_ia_new = seq_det(image=img, bounding_boxes=bboxes_ia)

        bboxes_new = []
        isegmaps_new = []
        for i in range(len(bboxes_ia)):
            if bboxes_ia_new[i].is_out_of_image(img.shape, fully=True, partly=False):
                bad_augment = True
                # BaseFewShotISEG.e_print('The bounding box is fully out of image after augmentation!')
                break
            bb: BoundingBox = bboxes_ia_new[i]
            bb.clip_out_of_image_(img.shape)
            box = np.array([bb.y1, bb.x1, bb.y2, bb.x2], dtype=np.float32)
            bboxes_new.append(box)

            cur_isegmap_ia_new = seq_det(segmentation_maps=isegmaps_ia[i:i + 1])
            arr = cur_isegmap_ia_new[0].get_arr()
            isegmaps_new.append(arr)
            del box

        if not bad_augment:
            img = img_new.astype(np.uint8)
            bboxes = np.array(bboxes_new, dtype=np.float32).reshape(-1, 4)
            isegmaps = np.array(isegmaps_new, dtype=np.bool)

        return img, bboxes, isegmaps

    def get_query(self,
                  idx,
                  idx_parent,
                  cat_id_main,
                  _cats_ids_to_sample_real: Optional[ndarray] = None,
                  path_only=False) -> Tuple[ndarray, ...]:
        img_fp = self.get_arb_img_fp_new(idx_parent)

        if path_only:
            return img_fp
        if self.batch > 1 and self.sampling_origin_ds not in ('OMNIISEG', 'MNISTISEG'):
            if self.ar_groups_order_fpaths_all[idx] != img_fp:
                print('bad')
                assert False
            assert self.ar_groups_order_fpaths_all[idx] == img_fp

        qry_img: ndarray = cv2.imread(img_fp, cv2.IMREAD_COLOR)[..., ::-1]

        # region Selection of categories as in FGN
        # Select one which is on image
        cats_on_img: list = list(self.qrys_parents_[idx_parent]['cats_dict'])
        assert cat_id_main in cats_on_img
        if _cats_ids_to_sample_real is None:
            cats_ids_to_sample_real: List[int] = [cat_id_main]
            # Select N - 1 cats randomly, cats may be not represented on the qry image
            # replace=False is required because different instances have to belong to different classes
            cats_to_save_bool = self.cats_to_save_bool.copy()
            cats_to_save_bool[cat_id_main] = 0
            # Do not select any of cats on query image
            if self.qry_cats_choice_remove:
                cats_to_save_bool[cats_on_img] = 0
            to_be_selected: List[int] = list(np.nonzero(cats_to_save_bool)[0])
            del cats_to_save_bool

            if self.qry_cats_choice_random:
                if len(to_be_selected) < self.n_ways - 1:
                    self.e_print(f'Could not select {self.n_ways} categories')
                    raise NotImplementedError
                selected_other = np.array(random.sample(to_be_selected, self.n_ways - 1))
            else:
                selected_other = to_be_selected[:(self.n_ways - 1)]
            cats_ids_to_sample_real.extend(selected_other)
            if self.qry_cats_order_shuffle:
                random.shuffle(cats_ids_to_sample_real)
            cats_ids_to_sample_real: ndarray = np.array(cats_ids_to_sample_real, dtype=np.int32)
            del to_be_selected, selected_other
        else:
            cats_ids_to_sample_real: ndarray = np.array(_cats_ids_to_sample_real, dtype=np.int32)
        # endregion

        qry_insts_ids: List = []
        qry_insts_cats_ids: List = []
        qry_insts_bboxes: List = []
        qry_insts_isegmaps: List = []
        for cat_id in cats_ids_to_sample_real:
            if cat_id not in cats_on_img:
                continue
            cat_id_insts_ids = self.qrys_parents_[idx_parent]['cats_dict'][cat_id]
            qry_insts_ids.extend(cat_id_insts_ids)
            for inst_id in cat_id_insts_ids:
                qry_insts_cats_ids.append(cat_id)
                bbox = self.insts[inst_id]['bbox']
                qry_insts_bboxes.append(bbox)
                info_isegmap = self.insts[inst_id]['info_isegmap']
                isegmap = self.get_isegmap(img=qry_img, bbox=bbox, info=info_isegmap)
                qry_insts_isegmaps.append(isegmap)
            del cat_id_insts_ids
        del cat_id

        qry_insts_ids: ndarray = np.array(qry_insts_ids, dtype=np.int32)
        qry_insts_cats_ids: ndarray = np.array(qry_insts_cats_ids, dtype=np.int32)
        qry_bboxes: ndarray = np.array(qry_insts_bboxes, dtype=np.float32)
        qry_isegmaps: ndarray = np.array(qry_insts_isegmaps, dtype=bool)

        h, w = qry_img.shape[:2]
        if self.upper_ds is not None and self.upper_ds.batch > 1:
            new_idx = idx
            original_idxs = np.where((self.upper_ds.order - len(self.upper_ds.order_initial))
                                     == new_idx)[0]
            unique = np.unique(original_idxs)
            # if len(unique) != 1:
            # print('Not unique values paths:')
            # print(self.upper_ds.ar_groups_order_fpaths_all[original_idxs])

            original_idx = unique[0]
            original_idx = int(original_idx)
            h_new, w_new = self.upper_ds.ar_group_new_hws[
                self.upper_ds.ar_groups_group_indexes_all[original_idx]]
            # print(f'New IDX={new_idx:04}, '
            #       f'Order {self.upper_ds.order[original_idx]} '
            #       f'Original IDX={original_idx:04}, '
            #       f'UpperDS-LEN', len(self.upper_ds),
            #       'Shape', h_new, w_new)
            # assert _h_new == h_new and _w_new == w_new
            # h_new, w_new = _h_new, _w_new
            if self.upper_ds.ar_groups_order_fpaths_all[original_idx] != img_fp:
                assert False
        elif self.batch == 1 or self.sampling_origin_ds in ('OMNIISEG', 'MNISTISEG'):
            h_new, w_new = get_new_shape(h, w, self.target_size, self.max_size)
            # h_new, w_new = (np.around(np.array([h_new, w_new]) / self.sub_sample_ratio)
            #                 * self.sub_sample_ratio).astype(np.int32)
        else:
            h_new, w_new = self.ar_group_new_hws[self.ar_groups_group_indexes_all[idx]]

        if h_new != h or w_new != w:
            qry_bboxes[:, [0, 2]] = qry_bboxes[:, [0, 2]] * (h_new / h)
            qry_bboxes[:, [1, 3]] = qry_bboxes[:, [1, 3]] * (w_new / w)

            qry_img: ndarray = cv2.resize(qry_img, (w_new, h_new))
            qry_isegmaps_new = []
            for i in range(len(qry_isegmaps)):
                # Forward conversion (to np.uint8) is required for OpenCV,
                # backwards - for faster copy to GPU.
                res = cv2.resize(qry_isegmaps[i].astype(np.uint8), (w_new, h_new)).astype(bool)
                qry_isegmaps_new.append(res)
            qry_isegmaps = np.array(qry_isegmaps_new, dtype=bool)

        if self.augment_qry:
            qry_img, qry_bboxes, qry_isegmaps = \
                self.augment_with_imgaug(self.augs_qry, qry_img, qry_bboxes, qry_isegmaps)

        return qry_img, qry_bboxes, qry_isegmaps, qry_insts_ids, qry_insts_cats_ids, cats_ids_to_sample_real

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
    def get_isegmaps_on_img_single(img: ndarray, isegmap: Union[ndarray]) -> SegmentationMapsOnImage:
        # Only one segmentation map due to the docs
        return SegmentationMapsOnImage(isegmap, shape=img.shape)

    @staticmethod
    def get_isegmaps_on_img_multiple(img: ndarray,
                                     isegmaps: Union[ndarray, List[ndarray]]) -> List[SegmentationMapsOnImage]:
        # Only one segmentation map due to the docs
        isegmaps = np.array(isegmaps, dtype=np.bool)
        h, w = isegmaps.shape[-2:]
        isegmaps = isegmaps.reshape((-1, h, w))
        res = []
        for i in range(len(isegmaps)):
            res.append(BaseFewShotISEG.get_isegmaps_on_img_single(img, isegmaps[i]))
        return res

    @staticmethod
    def draw_on_img(img: ndarray,
                    bboxes: Optional[Union[List[ndarray], ndarray, BoundingBoxesOnImage]] = None,
                    cat_ids: Optional[Union[List[Union[int, str, ndarray]], ndarray]] = None,
                    isegmaps: Optional[Union[List[ndarray], ndarray, SegmentationMapsOnImage]] = None,
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
            bbs = BaseFewShotISEG.get_bboxes_on_img_from_yxyx(img, bboxes, cat_ids, cat_id_comment)
            img = bbs.draw_on_image(img, color=color)

        if isegmaps is not None and len(isegmaps) > 0:
            for i in range(len(isegmaps)):
                iss = BaseFewShotISEG.get_isegmaps_on_img_single(img, isegmaps[i])
                img = iss.draw_on_image(img, colors=color, alpha=0.75)[0]

        return img

    @staticmethod
    def draw_on_img_and_show(img, boxes=None, cat_ids=None, isegmaps=None, cat_id_comment=''):
        img = BaseFewShotISEG.draw_on_img(img, cat_ids, boxes, isegmaps, cat_id_comment)
        plt.imshow(img)
        plt.show()

    @staticmethod
    def save(img, save_dir, img_name=None, img_name_comment=''):
        t = time_log_fancy()
        assert save_dir is not None
        if img_name is None:
            img_name = f'{t}-Image@{img_name_comment.strip()}.png'
        img_name = os.path.join(save_dir, img_name)
        BaseFewShotISEG.a_print(f'Saving the image to {img_name}')
        plt.imsave(img_name, img)
        plt.clf()

    @staticmethod
    def draw_on_img_and_save(img, save_dir, img_name=None, img_name_comment='',
                             boxes=None, cat_ids=None, isegmaps=None, cat_id_comment=''):
        img = BaseFewShotISEG.draw_on_img(img, boxes, cat_ids, isegmaps, cat_id_comment)
        BaseFewShotISEG.save(img, save_dir, img_name, img_name_comment)

    @staticmethod
    def cut_algorithm(ymin, ymax, h_offset, max_shape):
        ymin_cut = max(0, ymin - h_offset)
        ymin_box = ymin - ymin_cut
        ymax_box = ymax - ymin_cut
        ymax_cut = min(ymax + h_offset, max_shape)

        assert (ymax_box - ymin_box) == (ymax - ymin)
        return np.array([ymin_cut, ymin_box, ymax_cut, ymax_box], dtype=np.int32)

    @staticmethod
    def get_crop(img, ymin, xmin, ymax, xmax, h_offset, w_offset, crop_square=True, mode='reflect'):
        h = ymax - ymin
        w = xmax - xmin
        if h_offset == 0 and w_offset == 0:
            h_pad_down = h % 2
            w_pad_right = w % 2
            img = np.pad(img, [[0, h_pad_down], [0, w_pad_right], [0, 0]], mode=mode)
            crop = img[ymin:ymax + h_pad_down, xmin:xmax + w_pad_right]
            assert crop.shape[0] % 2 == 0 and crop.shape[1] % 2 == 0
            box = [0, 0, h, w]
        else:
            if crop_square:
                result_h = h + h_offset * 2
                result_h += result_h % 2
                result_w = w + w_offset * 2
                result_w += result_w % 2
                if result_h > result_w:
                    add_w_offset = (result_h - result_w) // 2
                    w_offset += add_w_offset
                elif result_w > result_h:
                    add_h_offset = (result_w - result_h) // 2
                    h_offset += add_h_offset
            else:
                mode = 'constant'

            h_pad_up = h_offset
            h_remainder = (h + h_offset * 2) % 2
            h_pad_down = h_offset + h_remainder
            w_pad_left = w_offset
            w_remainder = (w + w_offset * 2) % 2
            w_pad_right = w_offset + w_remainder
            img = np.pad(img, [[h_pad_up, h_pad_down], [w_pad_left, w_pad_right], [0, 0]], mode=mode)
            # crop = img[
            # ymin + h_pad_up - h_offset:ymax + h_pad_up + h_offset + h_remainder,
            # xmin + w_pad_left - w_offset:xmax + w_pad_left + w_offset + w_remainder
            # ]
            crop = img[ymin:ymax + h_pad_up + h_pad_down, xmin:xmax + w_pad_left + w_pad_right]
            box = [h_offset, w_offset, h_offset + h, w_offset + w]

        box = np.array(box, dtype=np.int32).reshape(4)
        return crop, box

    def get_support(self, qry_insts_ids, cats_ids_to_sample_real, spp_insts_ids=None) -> Tuple[ndarray, ...]:
        # Sampling k images for each category
        spp_imgs: List[ndarray] = []
        spp_bboxes: List[ndarray] = []
        spp_isegmaps: List[ndarray] = []
        spp_areas: List[float] = []

        assert len(cats_ids_to_sample_real) == self.n_ways, 'Too big amount of cats '

        if spp_insts_ids is None:
            spp_insts_ids: List[int] = []
            for cat_id in cats_ids_to_sample_real:
                insts_pool: Union[list, ndarray]
                if self.delete_qry_insts_in_spp_insts_on_train:
                    # During finetuning and testing there may be not enough instances
                    # because there is a limited amount of support examples per class only
                    # insts_pool = []
                    # cats_insts_list = self.cats_insts_list[cat_id].copy()
                    # random.shuffle(cats_insts_list)
                    # for v in self.cats_insts_list[cat_id]:
                    #     if v not in qry_insts_ids:
                    #         insts_pool.append(v)
                    #         if len(insts_pool) == self.k_shots:
                    #             break
                    insts_pool = [v for v in self.cats_insts_list[cat_id] if v not in qry_insts_ids]
                else:
                    insts_pool = self.cats_insts_list[cat_id]

                if self.spp_random:
                    if len(insts_pool) < self.k_shots:
                        self.e_print(f'Could not sample {self.k_shots} samples from {insts_pool}')
                        raise NotImplementedError
                    cat_insts_chosen = random.sample(insts_pool, self.k_shots)
                else:
                    cat_insts_chosen = insts_pool[:self.k_shots]
                spp_insts_ids.extend(cat_insts_chosen)
                del insts_pool, cat_insts_chosen
            spp_insts_ids: ndarray = np.array(spp_insts_ids)
            del cat_id

        # Read images with instances and crop ROI
        inst_id: int
        for inst_id in spp_insts_ids:
            img_id: int
            img: ndarray
            img_pad: ndarray
            img_crop: ndarray
            img_resized: ndarray
            cat_id: int
            bbox: Union[list, ndarray]
            info_isegmap: Union[list, ndarray]

            if self.finetune == 'Use' and self.insts[inst_id].get('num_parent_qry') is None:
                img_sp: str = self.insts[inst_id]['ft_img_sp']
                img_dir_fp: str = self.insts[inst_id]['ft_img_dir_fp']
                img_fp = os.path.join(img_dir_fp, img_sp)
            else:
                num_parent_qry = self.insts[inst_id]['num_parent_qry']
                img_fp = self.get_arb_img_fp_new(num_parent_qry)

            img = cv2.imread(img_fp, cv2.IMREAD_COLOR)[:, :, ::-1]
            cat_id = self.insts[inst_id]['cat_id']
            bbox = self.insts[inst_id]['bbox']
            info_isegmap = self.insts[inst_id]['info_isegmap']
            isegmap = self.get_isegmap(img=img, bbox=bbox, info=info_isegmap)

            ymin, xmin, ymax, xmax = np.array(bbox).astype(np.int32)

            # Warning: coordinates are np.float32
            w_offset = int(np.floor((xmax - xmin) * self.offset_ratio))
            h_offset = int(np.floor((ymax - ymin) * self.offset_ratio))

            # Note about types: bbs is the `imgaug` lib object
            # img* and isegmap* vars are ndarrays
            box_crop: ndarray
            bbs_crop: BoundingBoxesOnImage
            bbs_crop_resized: BoundingBoxesOnImage
            bbs_crop_pad: BoundingBoxesOnImage
            box_crop_pad: ndarray

            area = (ymax - ymin) * (xmax - xmin) / (img.shape[0] * img.shape[1])
            spp_areas.append(float(np.around(area, 3)))
            img_crop, box_crop = self.get_crop(img, ymin, xmin, ymax, xmax, h_offset, w_offset,
                                               crop_square=self.spp_crop_square)
            isegmap = np.expand_dims(isegmap, axis=2)
            isegmap_crop, _ = self.get_crop(isegmap, ymin, xmin, ymax, xmax, h_offset, w_offset,
                                            crop_square=self.spp_crop_square, mode='constant')
            isegmap_crop = np.squeeze(isegmap_crop, axis=2)
            del img, isegmap, w_offset, h_offset
            del ymin, xmin, ymax, xmax

            # --> Forward conversion
            bbs_crop = self.get_bboxes_on_img_from_yxyx(img_crop, box_crop, cat_id)

            img_crop_resized, bbs_crop_resized = self.resize_spp(image=img_crop, bounding_boxes=bbs_crop)
            isegmap_crop_resized, _ = self.resize_spp(image=isegmap_crop, bounding_boxes=bbs_crop)
            del img_crop, bbs_crop, isegmap_crop

            img_crop_pad, bbs_crop_pad = self.pad_spp(image=img_crop_resized, bounding_boxes=bbs_crop_resized)
            isegmap_crop_pad, _ = self.pad_spp(image=isegmap_crop_resized, bounding_boxes=bbs_crop_resized)
            del img_crop_resized, bbs_crop_resized, isegmap_crop_resized

            # <-- Backward conversion
            # There is only one bbox
            bbs_cur: BoundingBox = bbs_crop_pad[0]
            box_crop_pad = np.array([bbs_cur.y1, bbs_cur.x1, bbs_cur.y2, bbs_cur.x2], dtype=np.float32)

            box_crop_pad = box_crop_pad.reshape(4)
            if self.augment_spp:
                img_crop_pad, box_crop_pad, isegmap_crop_pad = \
                    self.augment_with_imgaug(self.augs_spp, img_crop_pad, box_crop_pad, isegmap_crop_pad)
            box_crop_pad = box_crop_pad.reshape(4)

            spp_imgs.append(img_crop_pad)
            spp_bboxes.append(box_crop_pad)
            spp_isegmaps.append(isegmap_crop_pad)
            del img_crop_pad, box_crop_pad, isegmap_crop_pad

        del cat_id, bbox, inst_id

        spp_imgs: ndarray = np.array(spp_imgs, dtype=np.uint8)
        spp_bboxes: ndarray = np.array(spp_bboxes, dtype=np.float32)
        spp_isegmaps: ndarray = np.array(spp_isegmaps, dtype=np.bool)
        spp_areas: ndarray = np.array(spp_areas, dtype=np.float32)
        return spp_imgs, spp_bboxes, spp_isegmaps, spp_insts_ids, spp_areas

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx: int, _qry_child_idx: int = None, _cats_ids_to_sample_real: int = None,
                    _spp_insts_ids: int = None, path_only=False):
        if self.overfit_sample is not None:
            return self.overfit_sample

        # Special merging case
        if self.merged_ds is not None:
            if idx >= len(self.order):
                idx -= len(self.order)
                # print(f'Sampling IDX={idx:04} from a merged ds of len', len(self.merged_ds))
                return self.merged_ds.__getitem__(idx, _qry_child_idx, _cats_ids_to_sample_real, _spp_insts_ids,
                                                  path_only)
            elif self.order[idx] >= len(self.order_initial):
                # h_new, w_new = self.ar_group_new_hws[self.ar_groups_group_indexes_all[idx]]
                original_idx = idx
                new_idx = self.order[original_idx] - len(self.order_initial)
                # print(f'Original IDX={original_idx:04}, Order {self.order[original_idx]}, '
                #       f'New IDX={new_idx:04}, '
                #       f'MergedDS-LEN', len(self.merged_ds),
                #       'Shape', h_new, w_new)
                return self.merged_ds.__getitem__(new_idx, _qry_child_idx, _cats_ids_to_sample_real, _spp_insts_ids,
                                                  path_only)

        if _qry_child_idx is None:
            real_idx = self.order[idx]
            if self.sampling_scenario == 'parents':
                qry_child_idx_all = self.qrys_parents_[real_idx]['nums_children_qrys']
                qry_child_idx = random.choice(qry_child_idx_all)
            else:
                qry_child_idx = real_idx
        else:
            qry_child_idx = _qry_child_idx

        # Parse to get the parent id
        if len(self.qrys_children) <= qry_child_idx:
            print('Some bad news')
            assert False
        idx_parent, cat_id_main = self.qrys_children[qry_child_idx]
        qry_img: ndarray
        qry_bboxes: ndarray
        qry_isegmaps: ndarray
        qry_insts_ids: ndarray
        qry_insts_cats_ids_real: ndarray
        cats_ids_to_sample_real: ndarray
        res = self.get_query(idx, idx_parent, cat_id_main, _cats_ids_to_sample_real, path_only=path_only)
        if path_only:
            path = res
            # noinspection PyTypeChecker
            assert isinstance(path, str)
            # noinspection PyTypeChecker
            assert os.path.exists(path)
            if self.overfit_sample_mode:
                self.overfit_sample = path
            return path

        qry_img, qry_bboxes, qry_isegmaps, qry_insts_ids, qry_insts_cats_ids_real, cats_ids_to_sample_real = res

        if _cats_ids_to_sample_real is not None:
            assert (cats_ids_to_sample_real == _cats_ids_to_sample_real).all()

        spp_imgs: ndarray
        spp_bboxes: ndarray
        spp_isegmaps: ndarray
        spp_insts_ids: ndarray
        spp_areas: ndarray
        spp_imgs, spp_bboxes, spp_isegmaps, spp_insts_ids, spp_areas = \
            self.get_support(qry_insts_ids, cats_ids_to_sample_real, _spp_insts_ids)
        if _spp_insts_ids is not None:
            assert (_spp_insts_ids == spp_insts_ids).all()

        # Create an array and fill the values
        mapping_array = np.zeros(np.max(cats_ids_to_sample_real) + 1, dtype=np.int32)
        mapping_array[cats_ids_to_sample_real] = np.arange(len(cats_ids_to_sample_real))
        qry_insts_cats_ids = mapping_array[qry_insts_cats_ids_real]
        cats_ids_to_sample = mapping_array[cats_ids_to_sample_real]

        sample = {
            'idx': idx,
            'qry_child_idx': qry_child_idx,
            'qry_img': qry_img,
            'qry_cat_ids_real': qry_insts_cats_ids_real.astype(np.int64),
            'qry_cat_ids': qry_insts_cats_ids.astype(np.int64),
            'qry_bboxes': qry_bboxes.astype(np.float32),
            'qry_isegmaps': qry_isegmaps.astype(np.bool),
            'spp_imgs': spp_imgs,
            'spp_bboxes': spp_bboxes.astype(np.float32),
            'spp_isegmaps': spp_isegmaps.astype(np.bool),
            'cats_ids_to_sample_real': cats_ids_to_sample_real.astype(np.int64),
            'cats_ids_to_sample': cats_ids_to_sample.astype(np.int64),
            'spp_insts_ids': spp_insts_ids.astype(np.int64),
            # Meta keys, shape before the transpose ops
            'img_shape': np.array(qry_img.shape, dtype=np.int32),
            'img_metas': [1],
            'spp_areas': spp_areas
        }

        if self.get_plot or self.overfit_sample_mode:
            sample['plot'] = self.visualize_item_pro(sample)

        if self.transforms:
            sample['qry_img']: Tensor = self.transforms(qry_img)
            shape = (self.n_ways * self.k_shots, 3, self.spp_img_size, self.spp_img_size)
            sample['spp_imgs']: Tensor = torch.zeros(size=shape, dtype=torch.float32)
            for i in range(len(spp_imgs)):
                sample['spp_imgs'][i] = self.transforms(spp_imgs[i])

        if self.overfit_sample_mode:
            self.overfit_sample = sample
            self.a_print('Sampled once for overfit and not sampling new')
        return sample

    def visualize_item_pro(self, item: Dict, with_isegmap=False) -> ndarray:
        qry_img: ndarray = item['qry_img']
        qry_bboxes: ndarray = item['qry_bboxes']
        qry_isegmaps: ndarray = item['qry_isegmaps']
        qry_cat_ids_real: ndarray = item['qry_cat_ids_real']
        qry_cat_ids: ndarray = item['qry_cat_ids']
        spp_imgs: ndarray = item['spp_imgs']
        spp_bboxes: ndarray = item['spp_bboxes']
        spp_isegmaps: ndarray = item['spp_isegmaps']
        cats_ids_to_sample: ndarray = item['cats_ids_to_sample']
        cats_ids_to_sample_real: ndarray = item['cats_ids_to_sample_real']
        spp_areas: ndarray = item['spp_areas']

        # Query image on the right, one column with query bboxes, 3x3 grid with support images on the left
        # 768 = 256 * 3 (3 rows, query is square),
        # 1792 = 768 (square query) + 256 (1 column) + 768 (3x3 grid)
        h = 768
        w = 768 + 256 + 256 * self.k_shots
        whole_image = np.zeros((h, w, 3), dtype=np.uint8) + 255

        # region Drawing the query
        left = 768
        right = left + 256

        cat_id_drawn = set()
        ext_cat_ids = []
        for i, cat_id in enumerate(qry_cat_ids):
            real_cat_id = qry_cat_ids_real[i]
            text: str = f'{cat_id}={real_cat_id}'
            ext_cat_ids.append(text)

            ymin, xmin, ymax, xmax = qry_bboxes[i].astype(np.int32)

            up = 256 * cat_id
            down = 256 * (cat_id + 1)
            # print(f'The image slice {ymin}:{ymax}, {xmin}:{xmax}')
            inst: ndarray = qry_img[ymin:ymax, xmin:xmax]
            isegmap = qry_isegmaps[i][ymin:ymax, xmin:xmax]
            # A custom way to draw an instance segmentation map
            if with_isegmap:
                inst = (inst * 0.75 +
                        np.expand_dims(isegmap * 255, axis=-1) * 0.5).astype(np.uint8)

            # Draw only one instance of cat in a grid in the middle of a plot
            if cat_id in cat_id_drawn:
                continue
            cat_id_drawn.add(cat_id)
            inst_res = self.resize_visualize(image=inst)
            inst_pad = self.pad_visualize(image=inst_res)
            inst_pad = cv2.putText(inst_pad, ext_cat_ids[i], (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150))
            whole_image[up:down, left:right] = inst_pad
            del inst, text

        bounding_boxes_on_image = self.get_bboxes_on_img_from_yxyx(qry_img, qry_bboxes, qry_cat_ids)
        # How it could be there is no bounding box for qry_img?
        qry_img = bounding_boxes_on_image.draw_on_image(qry_img)

        whole_image[:, :left] = cv2.resize(qry_img, (768, 768))
        # endregion

        for i in range(self.n_ways):
            cat_id = cats_ids_to_sample[i]
            real_cat_id = cats_ids_to_sample_real[i]
            up = i * 256
            down = (i + 1) * 256
            for j in range(self.k_shots):
                k = i * self.k_shots + j
                spp_img_now = spp_imgs[k]
                spp_isegmap_now = spp_isegmaps[k]
                # A custom way to draw an instance segmentation map
                if with_isegmap:
                    spp_img_now = (spp_img_now * 0.5 +
                                   np.expand_dims(spp_isegmap_now * 255, axis=-1) * 0.5).astype(np.uint8)

                spp_bbox_now = spp_bboxes[k]
                spp_area_now = spp_areas[k]

                text = f'{cat_id}={real_cat_id}@{spp_area_now:.03f}'
                boxes_on_image = self.get_bboxes_on_img_from_yxyx(img=spp_img_now, bboxes=spp_bbox_now, cat_ids=text)
                spp_img_now = boxes_on_image.draw_on_image(spp_img_now)
                spp_img_now = cv2.putText(spp_img_now, text, (5, 250),
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=0.5,
                                          color=(0, 255, 0),
                                          thickness=1,
                                          lineType=cv2.LINE_AA)

                left = 1024 + j * 256
                right = 1024 + (j + 1) * 256
                spp_img_now = self.resize_visualize(image=spp_img_now)
                whole_image[up: down, left: right] = spp_img_now

        whole_image[:, 768 + 256 * np.arange(1 + self.k_shots)] = [255, 0, 0]
        whole_image[256 * np.arange(3), 768:] = [255, 0, 0]
        return whole_image

    def write_state_dict(self):
        # Create something to track the last iterated element
        pass

    def load_state_dict(self):
        # Create something to track the load the iterated element
        pass

    def get_stats(self):
        stats = [
            ('Setup', self.setup),
            ('Base classes codes', self.cats_base_),
            ('Novel classes codes', self.cats_novel),
            ('Sampling dataset & subset', self.sampling_origin_ds + '-' + self.sampling_origin_ds_subset),
            ('Sampling classes', self.sampling_cats)
        ]
        for element in stats:
            self.v_print(f'{element[0]: <30}{element[1]}')

    def visualize(self, n_imgs=20, choose_random=False, action='save', vis_dir_sp: Optional[str] = None):
        assert action in ('save', 'show')

        if vis_dir_sp is not None:
            vis_dir_fp = os.path.join(self.root, vis_dir_sp)
        else:
            vis_dir_fp = os.path.join(self.root, self.suffix)
        assert isinstance(vis_dir_fp, str), f'Invalid path {vis_dir_fp}'

        if action == 'save':
            self.a_print(f'Saving visualize examples to the {vis_dir_fp}')
        create_empty_dir_unsafe(vis_dir_fp)
        old_transforms = self.transforms
        self.transforms = None
        merged_ds_old_transforms = old_transforms
        if self.merged_ds is not None:
            merged_ds_old_transforms = self.merged_ds.transforms
            self.merged_ds.transforms = None

        if n_imgs > len(self):
            self.e_print(f'Not enough samples visualization to choose from {len(self)} < {n_imgs}')
            n_imgs = len(self)
        if choose_random:
            samples = random.sample(list(range(len(self))), n_imgs)
        else:
            samples = np.arange(n_imgs)

        samples = sorted(samples)
        desc = f'Plotting {n_imgs} images from the dataset with len {self.__len__()}'

        for i, sample_idx in tqdm(enumerate(samples), desc=desc, total=n_imgs, ncols=120):
            item = self.__getitem__(idx=sample_idx)
            if self.merged_ds is not None and sample_idx >= len(self.order):
                img = self.merged_ds.visualize_item_pro(item)
            else:
                img = self.visualize_item_pro(item)

            if action == 'save':
                path = os.path.join(vis_dir_fp, f'Image {sample_idx:05}.png')
                plt.imsave(path, img)
                plt.close('all')
            else:
                plt.imshow(img)
                plt.show()

        self.transforms = old_transforms
        if self.merged_ds is not None:
            self.merged_ds.transforms = merged_ds_old_transforms

    def show_results_list_all_bboxes_no_isegmaps(self, results: List[Dict], save_dir_sp='evaluation_results'):
        save_dir_fp = os.path.join(self.root, save_dir_sp)
        create_empty_dir_unsafe(save_dir_fp)
        for num_res, result in enumerate(results):
            self.show_result_single_all_bboxes_no_isegmaps(result=result)

    def show_result_single_all_bboxes_no_isegmaps(self, result, num_res=0, save_dir_fp='evaluation_results'):
        create_empty_dir_safe(save_dir_fp)

        qry_img = result['qry_img']
        qry_bboxes = result['qry_bboxes']
        qry_isegmaps = result['qry_isegmaps']
        qry_cat_ids = result['qry_cat_ids']
        dt_scores = result['dt_scores']
        dt_bboxes = result['dt_bboxes']
        dt_cat_ids = result['dt_cat_ids']
        dt_isegmaps = result['dt_isegmaps']

        qry_img = self.denormalize(img=qry_img)
        name = f'Res.B.AllDetections.{num_res:05}.{self.suffix}.png'
        texts = [f'{int(dt_cat_ids[i])}~{float(dt_scores[i]):.02f}' for i in range(len(dt_cat_ids))]

        dt_img_cur = self.draw_on_img(qry_img, dt_bboxes, texts, isegmaps=None, color='r')
        dt_img_cur = cv2.putText(dt_img_cur.copy(),
                                 text=f'Res.{num_res}',
                                 org=(5, dt_img_cur.shape[0] - 5),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=1.0,
                                 color=(200, 100, 200),
                                 thickness=1)

        self.save(dt_img_cur, save_dir=save_dir_fp, img_name=name)

    def show_results_list_one_bbox_isegmap(self, results, save_dir_fp='evaluation_results'):
        create_empty_dir_unsafe(save_dir_fp)
        for num_res, result in enumerate(results):
            self.show_result_single_one_bbox_isegmap(result=result)

    def show_result_single_one_bbox_isegmap(self, result, num_res=0, save_dir_fp='evaluation_results'):
        create_empty_dir_safe(save_dir_fp)

        qry_img = result['qry_img']
        qry_img = self.denormalize(img=qry_img)
        qry_bboxes = result['qry_bboxes']
        qry_isegmaps = result['qry_isegmaps']
        qry_cat_ids = result['qry_cat_ids']
        dt_scores = result['dt_scores']
        dt_bboxes = result['dt_bboxes']
        dt_cat_ids = result['dt_cat_ids']
        ext_cat_ids = [f'{dt_cat_ids[i]}~{dt_scores[i]:.02f}' for i in range(len(dt_cat_ids))]
        dt_isegmaps = result['dt_isegmaps']

        for i in range(min(10, len(dt_bboxes))):
            name = f'Res.C.Det.{num_res:05}.Ex.{i:05}.{self.suffix}.png'
            if len(dt_isegmaps) != 0:
                dt_img_cur = self.draw_on_img(qry_img, [dt_bboxes[i]], [ext_cat_ids[i]], [dt_isegmaps[i]], color='r')
                cur_isegmap = np.dstack([dt_isegmaps[i]] * 3).astype(np.uint8) * 255
                dt_img_cur = np.column_stack((dt_img_cur, cur_isegmap))
            else:
                dt_img_cur = self.draw_on_img(qry_img, [dt_bboxes[i]], [dt_cat_ids[i]], color='r')
            dt_img_cur = cv2.putText(dt_img_cur.copy(),
                                     text=f'Res.{num_res}',
                                     org=(5, dt_img_cur.shape[0] - 5),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1.0,
                                     color=(255, 0, 0),
                                     thickness=1)
            self.save(dt_img_cur, save_dir=save_dir_fp, img_name=name)

    def evaluate(self, results: Optional[list] = None,
                 results_pkl_dir_fp: Optional[str] = None,
                 model_dir=None,
                 total: int = 1):
        """

        :param total:
        :param results:
        :param results_pkl_dir_fp: dir with saved pkl files, no redundant file
        :param model_dir:
        :return:
        """
        assert (results is not None) ^ (results_pkl_dir_fp is not None)
        print('Evaluating in the BaseFewShotISEG evaluate method ...')

        # Clear the evaluation results dir
        save_dir_sp = f'Results_{self.suffix}'
        if model_dir is not None:
            save_dir_fp = os.path.join(model_dir, save_dir_sp)
        else:
            save_dir_fp = os.path.join(self.root, save_dir_sp)
        del save_dir_sp
        create_empty_dir_unsafe(save_dir_fp)

        if results_pkl_dir_fp is not None:
            results_files_all = os.listdir(results_pkl_dir_fp)
            assert len(results_files_all) != 0
            results_files_one_sp = results_files_all[0]
            results_files_one_fp = os.path.join(results_pkl_dir_fp, results_files_one_sp)
            results = read_pkl(results_files_one_fp)

        rnd_inxs = np.random.choice(len(results), size=5, replace=False)

        for i in rnd_inxs:
            res: dict = results[i]
            res = res.copy()
            # Draw results on the image
            # Change sample['qry_img'] with a new image
            # Feed to visualize
            self.get_plot = True
            sample = self.__getitem__(
                idx=res['idx'],
                _qry_child_idx=res['qry_child_idx'],
                _cats_ids_to_sample_real=res['cats_ids_to_sample_real'],
                _spp_insts_ids=res['spp_insts_ids'])
            self.get_plot = False

            res['qry_img'] = sample['qry_img']
            res['qry_isegmaps'] = sample['qry_isegmaps']
            # res['qry_isegmaps'] = [decode(rle) for rle in res['qry_isegmaps_rle']]
            res['dt_isegmaps'] = [decode(rle) for rle in res['dt_isegmaps_rle']]

            whole_image = sample['plot']
            # Draw results over the whole image
            name = f'Res.A.Plot.{i:05}.{self.suffix}.png'
            self.save(whole_image, save_dir_fp, img_name=name)
            if len(res['dt_bboxes']) == 0:
                print('-> No detections for image', i)
                continue
            self.show_result_single_all_bboxes_no_isegmaps(res, i, save_dir_fp)
            self.show_result_single_one_bbox_isegmap(res, i, save_dir_fp)
            assert isinstance(res, dict)
        del results
        gc.collect()

        # Format dts results, remove background dts
        fsisegeval = FSISEGEval(results_pkl_dir_fp, n_ways=self.n_ways, iou_type='segm')
        with contextlib.redirect_stdout(None):
            fsisegeval.params.iouType = 'segm'
            fsisegeval.evaluate()
            fsisegeval.accumulate()
            eval_results_isegm: dict = fsisegeval.summarize_short()

            fsisegeval.params.iouType = 'bbox'
            fsisegeval.evaluate()
            fsisegeval.accumulate()
            eval_results_bbox: dict = fsisegeval.summarize_short()
        self.a_print(f'FSOD mAP {eval_results_bbox["mAP"]:.3f} & '
                     f'FSOD mAR {eval_results_bbox["mAR"]:.3f} & '
                     f'FSIS mAP {eval_results_isegm["mAP"]:.3f} & '
                     f'FSIS mAP {eval_results_isegm["mAR"]:.3f}')

        return {'isegm_mAP': eval_results_isegm['mAP'],
                'isegm_mAR': eval_results_isegm['mAR'],
                'bbox_mAP': eval_results_bbox['mAP'],
                'bbox_mAR': eval_results_bbox['mAR']}
