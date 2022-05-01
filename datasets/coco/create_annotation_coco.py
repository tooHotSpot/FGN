import os
from typing import List, Dict, Union

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imagesize

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from imgaug import SegmentationMapsOnImage
from imgaug import BoundingBox, BoundingBoxesOnImage

from cp_utils.cp_dir_file_ops import \
    define_env, \
    check_dir_if_exists, \
    check_file_if_exists, \
    create_empty_dir_safe, \
    read_json, \
    write_json_safe, \
    write_pkl_safe

if define_env() == 'PC':
    ds_root_path = 'D:/Datasets/COCO/'
elif define_env() == 'SERVER':
    ds_root_path = '/home/neo/Datasets/COCO'
elif define_env() == 'COLAB':
    ds_root_path = '/content/COCO'
else:
    raise NotImplementedError('Paths are not specified')


def get_mask_for_coco_img_inst(img_h, img_w, ann_segmentation):
    # Imported from pycocotools library
    if type(ann_segmentation) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(ann_segmentation, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif type(ann_segmentation['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(ann_segmentation, img_h, img_w)
    else:
        # rle
        rle = ann_segmentation

    m = maskUtils.decode(rle)
    return m


def create(subset='train2017'):
    # Old loading function
    # A much better loader is implemented in COCODS
    if not check_dir_if_exists('resources'):
        create_empty_dir_safe('resources')
    new_file_fp = f'resources/new_{subset}.json'
    if check_file_if_exists(new_file_fp):
        print('File already exists', new_file_fp)
        return

    subset_fp = os.path.join(ds_root_path, subset)
    ann_file_fp = os.path.join(ds_root_path, f'annotations/instances_{subset}.json')
    coco = COCO(ann_file_fp)

    result_json_data = {
        'ds_name': 'COCO',
        'subset': subset,
        'subset_dir_fp': subset_fp,
        'subset_isegmaps_dir_fp': '',
        'imgs': {
            'hs': [],
            'ws': [],
            'imgs_sps': [],
            # List of [[y1, x1, y2, x2], ann['category_id'], ann['segmentation']]
            'insts': []
        }
    }

    # Iterating over the list
    for img_inst in tqdm(coco.dataset['images']):
        img_sp = img_inst['file_name']
        height = img_inst['height']
        width = img_inst['width']
        annIds = coco.getAnnIds(imgIds=img_inst['id'])
        anns = coco.loadAnns(annIds)
        cur_insts_list = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h
            inst_list = [[y1, x1, y2, x2], ann['category_id'], ann['segmentation']]
            cur_insts_list.append(inst_list)

        result_json_data['imgs']['hs'].append(height)
        result_json_data['imgs']['ws'].append(width)
        result_json_data['imgs']['imgs_sps'].append(img_sp)
        result_json_data['imgs']['insts'].append(cur_insts_list)

    write_json_safe(new_file_fp, result_json_data)
    return


def from_json_to_pickle(subset='train2017'):
    json_file_fp = f'resources/new_{subset}.json'
    if not check_file_if_exists(json_file_fp):
        print('File does not exist', json_file_fp)
        return

    pkl_file_fp = f'resources/new_{subset}.pkl'
    data = read_json(json_file_fp)
    write_pkl_safe(pkl_file_fp, data)
    return


def show_annotations_on_img(img, bboxes, cat_ids, isegmaps, show=False):
    # Assuming that a bounding box has y1, x1, y2, x2 format

    bboxes_ia = []
    isegmaps_ia = []
    for i in range(len(bboxes)):
        bbox_cur = bboxes[i]
        y1, x1, y2, x2 = bbox_cur
        cat_ids_cur = None
        if cat_ids is not None:
            cat_ids_cur = cat_ids[i]
        bboxes_ia.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=cat_ids_cur))
        if isegmaps is not None:
            isegmaps_ia.append(SegmentationMapsOnImage(isegmaps[i], shape=img.shape))

    bboxes_on_img = BoundingBoxesOnImage(bboxes_ia, shape=img.shape)
    img = bboxes_on_img.draw_on_image(img)
    for i in range(len(isegmaps_ia)):
        img = isegmaps_ia[i].draw_on_image(img)[0]

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.show()
    return img


def filter_coco(coco_subset, name, show_stats=False, return_ann_small=False,
                ann_min_size_ratio: float = 0.005):
    print('Filtering the COCO subset')
    # + Better for training, since <<crowd>> annotations result in many RPN positive boxes
    # + Even with <<crowd>> annotation fed to support, in few-shot scenario it is OK
    # - On evaluation, <<crowd>> annotation may be 1 FN and many FP
    # ! True few-shot testing is performed with no crowd annotations in the support
    # ! Since <<crowds>> are present in COCO only, the problem is not important when we use VOC for test
    # ! <<crowd>> annotations are present on a truly low ratio of images in the whole dataset
    # and even lower ratio in a whole set of annotations
    print('Important notice: crowd annotations are not discarded in this implementation!')

    filtered_imgs_with_anns_dict: List[Dict[str, Union[int, List[int]]]] = []
    # Counters
    false_hw_imgs: int = 0
    anns_ids_ttl: int = 0
    anns_ids_valid_ttl: int = 0

    crowd_anns_ttl: int = 0
    crowd_anns_imgs_ttl: int = 0

    ignored_anns_ttl: int = 0
    ignored_anns_imgs_ttl: int = 0

    bad_anns_ttl: int = 0
    bad_anns_imgs_ttl: int = 0

    small_anns_ttl: int = 0
    small_anns_imgs_ttl: int = 0
    # May be required for  visualization of these small annotations
    small_anns_cat_ids: List[int] = []
    small_anns_img_ids: List[int] = []
    small_anns_ann_ids: List[int] = []

    imgs_ids_sorted = sorted(coco_subset.imgs)
    for img_id in tqdm(imgs_ids_sorted):
        crowd_ann_img: bool = False
        ignored_ann_img: bool = False
        bad_ann_img: bool = False
        small_ann_img: bool = False
        # Rescale
        img_info = coco_subset.imgs[img_id]
        img_h, img_w = img_info['height'], img_info['width']
        # Rule 0
        # Check image real size and the one in the annotation
        img_fp = os.path.join(ds_root_path, name, img_info['file_name'])
        real_width, real_height = imagesize.get(img_fp)
        if real_width != img_w or real_height != img_h:
            false_hw_imgs += 1
            if false_hw_imgs == 1:
                print('Example of false HW image')
                print('Image', img_fp, 'Real', real_height, real_width, 'Ann', img_h, img_w)
                # Using assert till encounter an example and find out how to handle it
                assert False

        def ann_print(t_img_id: int, t_ann: dict, comment: str):
            t_ann = t_ann.copy()
            print('')
            print(comment)
            print('Image info', coco_subset.imgs[t_img_id])
            # Delete <segmentation> key and not print it
            t_ann.pop('segmentation', None)
            print('Annotation:', t_ann)

        # Rescaled
        ann_ids = coco_subset.getAnnIds(imgIds=[img_id])
        anns = coco_subset.loadAnns(ann_ids)
        ann_ids_valid = []
        for ann_id, ann in zip(ann_ids, anns):
            anns_ids_ttl += 1
            crowd_ann: bool = False
            ignored_ann: bool = False
            bad_ann: bool = False
            small_ann: bool = False
            # Rule 1 (Used just for counting)
            if ann['iscrowd']:
                if not crowd_ann_img:
                    crowd_ann_img = True
                    crowd_anns_imgs_ttl += 1
                    if crowd_anns_imgs_ttl == 1:
                        ann_print(img_id, ann, comment='Example of a crowd annotation')
                crowd_ann = True
                crowd_anns_ttl += 1
            # Rule 2
            if ann.get('ignore', False):
                if not ignored_ann_img:
                    ignored_ann_img = True
                    ignored_anns_imgs_ttl += 1
                    if ignored_anns_imgs_ttl == 1:
                        ann_print(img_id, ann, comment='Example of an ignored annotation')
                ignored_ann = True
                ignored_anns_ttl += 1
            # Rule 3
            # Rounding and casting are the same to COCODS
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h
            y1, x1, y2, x2 = np.around([y1, x1, y2, x2]).astype(np.int32)
            ann_h = y2 - y1
            ann_w = x2 - x1
            ratio = (ann_h * ann_w) / (img_h * img_w)
            if ratio < ann_min_size_ratio:
                if not small_ann_img:
                    small_ann_img = True
                    small_anns_imgs_ttl += 1
                    if small_anns_imgs_ttl == 1:
                        ann_print(img_id, ann, comment='Example of a small annotation')
                        print(f'Image h={img_h} w={img_w} area={img_h * img_w} '
                              f'Ann.  h={ann_h} w={ann_w} area={ann_h * ann_w} '
                              f'Ratio {ann_h * ann_w} / {img_h * img_w} = {ratio:.03f}')
                small_ann = True
                small_anns_ttl += 1
                small_anns_img_ids.append(img_id)
                small_anns_ann_ids.append(ann_id)
                small_anns_cat_ids.append(ann['category_id'])

            # Rule 4
            if ann['area'] <= 0 or w < 1 or h < 1:
                if not bad_ann_img:
                    bad_ann_img = True
                    bad_anns_imgs_ttl += 1
                    if bad_anns_imgs_ttl == 1:
                        ann_print(img_id, ann, comment='Example of a bad annotation')
                bad_ann = True
                bad_anns_ttl += 1
            # Rule 6
            # Add a valid annotation to current image annotations
            # Flag <<crowd_ann>> is removed from the list
            # Flag <<small_ann>> is removed from the list
            if any((ignored_ann, bad_ann)):
                continue
            else:
                ann_ids_valid.append(ann_id)
                anns_ids_valid_ttl += 1
            # Additional possible rules
            # Check segmentation mask lies inside the bounding box
            # Other
        # Adding images with valid annotations only
        if len(ann_ids_valid) == 0:
            continue
        filtered_imgs_with_anns_dict.append(
            {'img_id': img_id, 'ann_ids_valid': ann_ids_valid}
        )
        del img_id, ann_ids_valid

    if show_stats:
        print(f'#Images initial  {len(coco_subset.imgs):05}')
        print(f'#Images final    {len(filtered_imgs_with_anns_dict):05}')
        print(f'#Annot. initial  {anns_ids_ttl:05}')
        print(f'#Annot. final    {anns_ids_valid_ttl:05}')
        print(f'-> false_hw_imgs {false_hw_imgs:05}')
        print(f'-> crowd_anns    {crowd_anns_ttl:05} #imgs {crowd_anns_imgs_ttl:05}')
        print(f'-> ignored_anns  {ignored_anns_ttl:05} #imgs {ignored_anns_imgs_ttl:05}')
        print(f'-> bad_ann       {bad_anns_ttl:05} #imgs {bad_anns_imgs_ttl:05}')
        print(f'-> ann_small     {small_anns_ttl:05} #imgs {small_anns_imgs_ttl:05}')
        print('Important notice: <<crowd>> annotations are not discarded in this implementation!')

        ann_small_dict = {coco_subset.cats[cat_id]['name']: 0
                          for cat_id in set(small_anns_cat_ids)}
        for cat_id in small_anns_cat_ids:
            name = coco_subset.cats[cat_id]['name']
            ann_small_dict[name] += 1
        print(f'Distribution of objects smaller than {ann_min_size_ratio} in h/w on a final image')
        names = np.array(sorted(ann_small_dict))
        values = np.array([ann_small_dict[n] for n in names])
        order = np.argsort(values)[::-1]
        # for i in order:
        #     print(f'name {names[i]:10} => value {values[i]:05}')
        print(list(zip(names[order], values[order])))

    if return_ann_small:
        small_anns_img_ids: np.ndarray = np.array(small_anns_img_ids, dtype=np.int32)
        small_anns_ann_ids: np.ndarray = np.array(small_anns_ann_ids, dtype=np.int32)
        small_anns_cat_ids: np.ndarray = np.array(small_anns_cat_ids, dtype=np.int32)
        return filtered_imgs_with_anns_dict, small_anns_ann_ids, small_anns_img_ids, small_anns_cat_ids

    return filtered_imgs_with_anns_dict


if __name__ == '__main__':
    # create(subset='val2017')
    create(subset='train2017')
    # from_json_to_pickle(subset='val2017')
    # from_json_to_pickle(subset='train2017')
