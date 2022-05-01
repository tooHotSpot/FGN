import gc
import os
from tqdm import tqdm

import numpy as np
from numpy import ndarray
from typing import List, Union, Dict
import imgaug.augmenters as iaa

from pycocotools import coco
from pycocotools import mask as maskUtils
from datasets.coco.create_annotation_coco import filter_coco

from cp_utils.cp_dir_file_ops import define_env, read_pkl, write_pkl_safe
from cp_utils.cp_dir_file_ops import read_json, write_json_safe
from cp_utils.cp_dir_file_ops import check_file_if_exists
from datasets.mnistiseg.mnistiseg_ds import MNISTISEG, example
from datasets.fewshotiseg.base_fst import BaseFewShotISEG

if define_env() == 'PC':
    ds_root_path = 'D:/Datasets/COCO/'
elif define_env() == 'SERVER':
    ds_root_path = '/home/neo/Datasets/COCO'
elif define_env() == 'COLAB':
    ds_root_path = '/content/COCO'
else:
    raise NotImplementedError('Paths are not specified')


class COCODS(MNISTISEG):
    # Reduce from (800, 1333) to (500, 800)
    # Reduce the training time
    # Reduce the task complexity for RPN in few-shot mode
    target_size = 800
    max_size = 1333
    # There are only train2017 and val2017 COCO data sets available in 2021
    imgs_set_possible = ('train', 'val')

    # ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    root = '../../datasets/coco/resources'
    samples_dir = 'coco_samples'

    augs_seq = iaa.Sequential([
        iaa.SomeOf(1, [
            iaa.HorizontalFlip(p=0.5),
            # iaa.Crop(percent=(0, 0.05), keep_size=True, sample_independently=True),
        ]),
        iaa.SomeOf(1, [
            # iaa.Grayscale(),
            iaa.AddToHue(value=(-10, 10)),
            iaa.AddToSaturation(value=(-25, 25)),
            iaa.AddToBrightness(add=(-30, 30)),
        ]),
        iaa.SomeOf(1, [
            iaa.GaussianBlur(sigma=(0.0, 1.5)),
        ]),
    ])

    info_isegmaps_counts: List[List[bytes]] = []
    info_isegmaps_hw_sizes: List[ndarray] = []
    # Code development flags
    is_list = 0
    has_counts = 0
    simple = 0

    original_to_new_cats_ids: ndarray
    new_cats_ids_to_original: ndarray
    cats_names_to_new_cats_ids: Dict[str, int]
    new_cats_ids_to_cats_names: Dict[int, str]

    def __init__(self, **kwargs):
        file_sp = 'COCOCats.json'
        file_fp = os.path.join(self.root, file_sp)
        if not os.path.exists(file_fp):
            ann_file = os.path.join(ds_root_path, 'annotations', 'instances_val2017.json')
            assert os.path.exists(ann_file), f'Annotation file {ann_file} does not exists'
            coco_cur = coco.COCO(ann_file)
            coco_cats = coco_cur.cats
            write_json_safe(file_fp, coco_cats)
        else:
            coco_cats = read_json(file_fp)

        # Minus 1 for debug purpose, will raise error in bad indexing
        cats_sorted = np.sort(np.array(list(coco_cats.keys()), dtype=np.int32))
        self.original_to_new_cats_ids = np.zeros(cats_sorted[-1] + 1, dtype=np.int32) - 1
        self.new_cats_ids_to_original = cats_sorted
        self.original_to_new_cats_ids[cats_sorted] = np.arange(len(cats_sorted))

        self.cats_names_to_new_cats_ids = dict()
        self.new_cats_ids_to_cats_names = dict()
        for cat_id in coco_cats:
            original_id = coco_cats[cat_id]['id']
            name = coco_cats[cat_id]['name']
            if int(original_id) != int(cat_id):
                print(f'Not equal cat ids:'
                      f'from dict entry {original_id}, '
                      f'from dict index {cat_id}, '
                      f'name from dict entry {name}')
            new_cat_id = self.original_to_new_cats_ids[int(cat_id)]
            self.cats_names_to_new_cats_ids[name] = new_cat_id
            self.new_cats_ids_to_cats_names[new_cat_id] = name

        super(COCODS, self).__init__(**kwargs)
        # For COCO has to be set manually for simple DS and few-shot DS
        imgs_set = self.imgs_set + '2017'
        self.imgs_dir_fp = os.path.join(ds_root_path, imgs_set)

    def check_all_files_exist(self):
        files_sps = ['_imgs_sps.pkl', '_rles_counts.pkl',
                     '_bboxes.pkl', '_cat_ids.pkl', '_rles_hw_sizes.pkl']
        imgs_set = self.imgs_set + '2017'
        for f_sp in files_sps:
            full_f_fp = os.path.join(self.root, f'{imgs_set}{f_sp}')
            if not check_file_if_exists(full_f_fp):
                print('File not found:', full_f_fp)
                return False
        return True

    def write_all_files(self):
        # Save to files
        imgs_set = self.imgs_set + '2017'
        write_pkl_safe(os.path.join(self.root, f'{imgs_set}_imgs_sps.pkl'), self.imgs_sps)
        write_pkl_safe(os.path.join(self.root, f'{imgs_set}_rles_counts.pkl'), self.info_isegmaps_counts)
        # Result _pickle files are smaller result np.{save/load} files
        write_pkl_safe(os.path.join(self.root, f'{imgs_set}_bboxes.pkl'), self.bboxes)
        write_pkl_safe(os.path.join(self.root, f'{imgs_set}_cat_ids.pkl'), self.cat_ids)
        write_pkl_safe(os.path.join(self.root, f'{imgs_set}_rles_hw_sizes.pkl'), self.info_isegmaps_hw_sizes)

    def read_all_files(self):
        imgs_set = self.imgs_set + '2017'
        self.imgs_sps = read_pkl(os.path.join(self.root, f'{imgs_set}_imgs_sps.pkl'))
        self.info_isegmaps_counts = read_pkl(os.path.join(self.root, f'{imgs_set}_rles_counts.pkl'))
        self.bboxes = read_pkl(os.path.join(self.root, f'{imgs_set}_bboxes.pkl'))
        self.cat_ids = read_pkl(os.path.join(self.root, f'{imgs_set}_cat_ids.pkl'))
        self.info_isegmaps_hw_sizes = read_pkl(os.path.join(self.root, f'{imgs_set}_rles_hw_sizes.pkl'))

    def read_data(self):
        # Reading COCO data this way is much faster than conventional COCO(annotation_file)
        # because of data split and efficient formats.
        # - BBoxes are converted from float to np.int32.
        # - Data is split into numerical, string and bytes
        # - Numerical data is stored in np.int16/np.uint8 formats with np.load/save. Be attentive with that.
        # - Total annotation data for train2017 is reduced >10X (from 450 MB to 28 MB)

        imgs_set = self.imgs_set + '2017'
        self.imgs_dir_fp = os.path.join(ds_root_path, imgs_set)
        self.imgs_sps = []
        self.bboxes = []
        self.cat_ids = []
        self.info_isegmaps = []

        self.info_isegmaps_counts = []
        self.info_isegmaps_hw_sizes = []

        # Reading all the data from several files. May be unified, but different formats
        if not self.check_all_files_exist():
            print('COCODS: creating a new annotation file')
            # Collect COCO data into a more lightweight chunks
            # 1. Load COCO data
            ann_file_fp = os.path.join(ds_root_path, f'annotations/instances_{imgs_set}.json')
            coco_subset = coco.COCO(annotation_file=ann_file_fp)

            # 2. Filter it
            filtered_imgs_with_anns_dict = filter_coco(coco_subset, imgs_set, show_stats=True)
            print(f'Filtering: reduced amount of data from '
                  f'{len(coco_subset.imgs)} to {len(filtered_imgs_with_anns_dict)}')

            # 3. Gather it to new lists (with an additional check on the segmentation)
            for data in tqdm(filtered_imgs_with_anns_dict, 'Loop over filtered images'):
                img_id = data['img_id']
                # An interface to load all the annotations IDS for this image
                # ann_ids = coco_subset.getAnnIds(imgIds=[img_id])
                ann_ids_valid = data['ann_ids_valid']
                img_info = coco_subset.imgs[img_id]
                img_sp = img_info['file_name']
                self.imgs_sps.append(img_sp)
                img_h, img_w = img_info['height'], img_info['width']
                anns = coco_subset.loadAnns(ann_ids_valid)

                this_bboxes = []
                this_cat_ids = []
                this_info_isegmaps_counts = []
                this_info_isegmaps_hw_sizes = []
                for ann_id, ann in zip(ann_ids_valid, anns):
                    # Instances with width or height less than ann_min_size are deleted during filtration
                    x1, y1, w, h = ann['bbox']
                    x2 = x1 + w
                    y2 = y1 + h
                    this_bboxes.append(np.around([y1, x1, y2, x2]).astype(np.int32))
                    original_cat_id = ann['category_id']
                    new_cat_id = self.original_to_new_cats_ids[original_cat_id]
                    this_cat_ids.append(new_cat_id)
                    ann_segmentation = ann['segmentation']
                    rle = self.get_rle_for_coco_img_inst(img_h, img_w, ann_segmentation)
                    assert len(rle) == 2 and 'counts' in rle and 'size' in rle
                    assert isinstance(rle['counts'], bytes)
                    rle_list = [rle['size'], rle['counts']]
                    m = self.get_isegmap(img=None, bbox=this_bboxes[-1], info=rle_list)
                    # Check
                    y1, x1, y2, x2 = this_bboxes[-1]
                    if np.sum(m[y1:y2, x1:x2]) == 0:
                        BaseFewShotISEG.e_print('Mask for COCO instance decoded from RLE is empty')
                    del m
                    this_info_isegmaps_counts.append(rle['counts'])
                    this_info_isegmaps_hw_sizes.append(rle['size'])
                # Original bbox coordinated in the COCO have a float dtype
                # Solution: around and compress to np.int16 array, not np.uint16 to omit problems
                # with subtraction overflow
                this_bboxes = np.array(this_bboxes).astype(np.int16)
                # Compress cat_ids to uint8 array
                this_cat_ids = np.array(this_cat_ids).astype(np.uint8)
                self.bboxes.append(this_bboxes)
                self.cat_ids.append(this_cat_ids)
                self.info_isegmaps_counts.append(this_info_isegmaps_counts)
                this_info_isegmaps_hw_sizes = np.array(this_info_isegmaps_hw_sizes).astype(np.int16)
                self.info_isegmaps_hw_sizes.append(this_info_isegmaps_hw_sizes)

            self.write_all_files()
        else:
            self.read_all_files()

        # Important note: backward cast from np.int16 to np.int32/np.float32
        # is performed in base classes __getitem__() methods
        print(f'RLE generation flags: '
              f'is_list={self.is_list}, has_counts={self.has_counts}, simple={self.simple}')

        for i in range(len(self.info_isegmaps_counts)):
            cur_info = []
            for j in range(len(self.info_isegmaps_counts[i])):
                size = self.info_isegmaps_hw_sizes[i][j]
                counts = self.info_isegmaps_counts[i][j]
                # Conventional form for maskUtils.decode method: dict
                # cur_info.append({'size': size, 'counts': counts})
                # Proposed form: tuple or list. Vital: size may be np.int16 dtype.
                cur_info.append([size, counts])
            self.info_isegmaps.append(cur_info)
        print('Finished with reading COCO data')
        self.len = len(self.imgs_sps)
        self.first_only = len(self.imgs_sps)
        del self.info_isegmaps_hw_sizes, self.info_isegmaps_counts
        gc.collect()
        return

    def get_rle_for_coco_img_inst(self, img_h, img_w, ann_segmentation):
        # Imported from pycocotools library
        if type(ann_segmentation) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(ann_segmentation, img_h, img_w)
            rle = maskUtils.merge(rles)
            self.is_list += 1
        elif type(ann_segmentation['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(ann_segmentation, img_h, img_w)
            self.has_counts += 1
        else:
            # rle
            rle = ann_segmentation
            self.simple += 1

        return rle

    @staticmethod
    def get_isegmap(img: None, bbox: Union[list, ndarray], info: Union[bytes, str, list]) -> ndarray:
        # info_rle is changed to info arg
        # Conventionally, info_rle may be a dict with <size> and <counts> fields
        # Also, it may be a list with 1st element <size> (of np.int16) and second
        if not isinstance(info, dict):
            # Conversion from np.int16 to np.int32 required
            h, w = info[0]
            size = [int(h), int(w)]
            counts = info[1]
            info = {'size': size, 'counts': counts}

        m = maskUtils.decode(info)
        return m

    def count_mean_std(self, n_imgs=1000):
        print('Using ImageNet params')
        return


if __name__ == '__main__':
    ds = COCODS(imgs_set='train', augment=True)
    print(ds.mean)
    print(ds.std)
    # example(ds)
    ds.visualize(n_imgs=50, with_isegmaps=False, action='save')
