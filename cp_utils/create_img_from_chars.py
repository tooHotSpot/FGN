import os
from tqdm import tqdm

import cv2
import numpy as np
import random

import torch
from torchvision.ops import box_iou

from cp_utils.cp_dir_file_ops import create_empty_dir_unsafe, write_pkl_unsafe

# Optimization: change to yield during `while True` loop
PALETTE = []
PALETTE_NP = []
ELEMENT = None


def cut_char_img(img, shift=1):
    if len(img.shape) == 3:
        # Select only one dimension to operate in grayscale
        img = img[..., 0]

    ymin_line = np.min(img, axis=1)
    indices_y = np.where(ymin_line != 255)[0]
    ymin, ymax = indices_y[0], indices_y[-1]
    ymin = max(0, ymin - shift)
    ymax = min(ymax + shift, img.shape[0])
    del ymin_line, indices_y

    xmin_line = np.min(img, axis=0)
    indices_x = np.where(xmin_line != 255)[0]
    xmin, xmax = indices_x[0], indices_x[-1]
    xmin = max(0, xmin - shift)
    xmax = min(xmax + shift, img.shape[1])
    del xmin_line, indices_x

    img_cut = img[ymin:ymax, xmin:xmax]
    return img_cut


def get_palette_np():
    global PALETTE
    global PALETTE_NP
    if len(PALETTE) == 0:
        values = [0, 0.5, 1.0]
        from itertools import product
        arr = np.array(list(product(values, repeat=3)), dtype=np.float32)
        # Remove the last (255, 255, 255)
        arr = arr[:-1]
        PALETTE_NP = (arr * 255).astype(np.uint8)
        PALETTE = PALETTE_NP.tolist()
        print('Computed PALETTE once')
    return PALETTE_NP


def get_palette_list():
    global PALETTE
    if len(PALETTE) == 0:
        get_palette_np()
    return PALETTE


def resize_char_img(img, min_max_ratios):
    h_cut, w_cut = img.shape[:2]
    # Float values are also possible
    ratio = random.uniform(min_max_ratios[0], min_max_ratios[1])
    h_cut, w_cut = int(h_cut * ratio), int(w_cut * ratio)
    img_res = cv2.resize(img, (w_cut, h_cut))
    return img_res


def paste_colored_char_img(img, img_cut_char, bboxes, colors, iou_max=0.25, hw_max=256):
    h_cut_res, w_cut_res = img_cut_char.shape[:2]

    attempts = 0
    while True:
        new_ymin = np.random.randint(0, hw_max - h_cut_res)
        new_ymax = new_ymin + h_cut_res
        new_xmin = np.random.randint(0, hw_max - w_cut_res)
        new_xmax = new_xmin + w_cut_res
        bbox = np.array([new_ymin, new_xmin, new_ymax, new_xmax])

        if len(bboxes) == 0:
            break
        else:
            ious = box_iou(torch.from_numpy(bboxes),
                           torch.from_numpy(np.expand_dims(bbox, 0)))
            if max(ious) < iou_max:
                break
        attempts += 1
        if attempts == 50:
            # Could not find a place
            return False

    if len(colors) != 0:
        available = np.ones(len(get_palette_np()))
        for c in colors:
            index = np.nonzero(np.product(c == get_palette_np(), axis=-1))
            available[index] = 0
        available = np.nonzero(available)[0]
        idx = np.random.choice(available)
    else:
        idx = np.random.choice(len(get_palette_np()))
    color = get_palette_np()[idx]

    img_cut_res_inverted = (255 - img_cut_char)
    img_cut_res_colored = np.dstack(tuple([img_cut_res_inverted] * 3))
    # Convert to float
    img_cut_res_colored = img_cut_res_colored.astype(np.float32)
    img_cut_res_colored = img_cut_res_colored * (1 - color / 255)
    img_cut_res_colored = 255.0 - img_cut_res_colored
    # Converted back to uint8
    img_cut_res_colored = img_cut_res_colored.astype(np.uint8)

    # Initial way, leads to high-contrast spots because of the overflow
    # img_cut_res_colored[img_cut_res_colored == 255] = 0
    # img[new_ymin:new_ymax, new_xmin:new_xmax] += img_cut_res_colored
    # Current way. Taking z coordinates into account works as alpha channel,
    # which results in a much different effect from a real object overlay.
    x, y, z = np.where(img_cut_res_colored < 245)
    del z
    img[new_ymin:new_ymax, new_xmin:new_xmax][x, y] = \
        img_cut_res_colored[x, y]

    if len(bboxes) != 0:
        bboxes = np.row_stack((bboxes, bbox))
        colors = np.row_stack((colors, color))
    else:
        bboxes = np.array([bbox])
        colors = np.array([color])

    return img, bboxes, colors


def get_char_mask_by_color(img, bbox, color, color_int_shift=75):
    arr = np.zeros(img.shape[:2], dtype=np.uint8)
    ymin, xmin, ymax, xmax = bbox
    roi = img[ymin:ymax, xmin:xmax]

    color_float = color.astype(np.float32)
    color_max = np.minimum(color_float + color_int_shift, 255)
    color_min = np.maximum(0, color_float - color_int_shift)
    rmax, gmax, bmax = color_max.astype(np.uint8)
    rmin, gmin, bmin = color_min.astype(np.uint8)

    # Optimization: find and appropriate numpy function and/or use simultaneous check
    mask_from_roi = (rmin <= roi[..., 0]) * (roi[..., 0] <= rmax) * \
                    (gmin <= roi[..., 1]) * (roi[..., 1] <= gmax) * \
                    (bmin <= roi[..., 2]) * (roi[..., 2] <= bmax)

    mask_from_roi = np.array(mask_from_roi, dtype=np.uint8)
    global ELEMENT
    if ELEMENT is None:
        ELEMENT = np.ones((3, 3), dtype=np.uint8)
    mask_from_roi = cv2.dilate(mask_from_roi, ELEMENT)
    arr[ymin:ymax, xmin:xmax] = mask_from_roi
    return arr


def create_ds(img_new_set_origin_imgs, new_subset_quantities,
              sizes_max_amount, sizes_min_max_ratios,
              result_imgs_root_fp, img_new_size, cat_id_in_upper_dir=False):
    """
    - Given a set of parameters, create a synthetic dataset which
    has an almost random amount of objects (e.g. letters) of different sizes
    a little bit cluttered.
    - cat_ids, boxes and segmentation maps complement images.
    - Images may be created on the fly, but this requires reading a lot of images
    at the same time.
    - Also, there is an already published Cluttered Omniglot.
    """

    for subset in new_subset_quantities:
        counter = 0
        bboxes = []
        cat_ids = []
        colors = []

        subset_quantity = new_subset_quantities[subset]
        result_imgs_subset_fp = os.path.join(result_imgs_root_fp, subset)
        create_empty_dir_unsafe(result_imgs_subset_fp)
        progress_bar = tqdm(total=subset_quantity, desc='While generator loop')
        while counter < subset_quantity:
            img_new_path = os.path.join(result_imgs_subset_fp, "%06d.jpg" % counter)
            new_img = np.ones((img_new_size, img_new_size, 3), dtype=np.int32) * 255

            bboxes_cur = []
            cat_ids_cur = []
            colors_cur = []

            # Size could be large, medium and small
            for size in sorted(sizes_max_amount.keys()):
                n = np.random.randint(0, sizes_max_amount[size])
                cur_min_max_ratios = sizes_min_max_ratios[size]
                for _ in range(n):
                    idx = np.random.randint(0, len(img_new_set_origin_imgs[subset]))
                    img_fp = img_new_set_origin_imgs[subset][idx]
                    img = cv2.imread(img_fp)[..., ::-1]
                    # Assume that img is a black sign on a white background
                    # Cut the sign of the image
                    img_cut = cut_char_img(img)
                    # Resize it
                    img_cut_res = resize_char_img(img_cut, cur_min_max_ratios)
                    # Paste it
                    result = paste_colored_char_img(new_img, img_cut_res, bboxes_cur, colors_cur,
                                                    iou_max=0.2, hw_max=img_new_size)
                    if result:
                        new_img = result[0]
                        bboxes_cur = result[1]
                        colors_cur = result[2]
                        upper_dir_fp, img_sp = os.path.split(img_fp)
                        if cat_id_in_upper_dir:
                            # For Omniglot
                            _, cat_id_dir_sp = os.path.split(upper_dir_fp)
                            cat_id = int(cat_id_dir_sp[-2:]) - 1
                        else:
                            # For MNIST
                            cat_id = int(img_sp[0])
                        cat_ids_cur.append(cat_id)
                        del cat_id

                if len(bboxes_cur) > 4:
                    break

            if len(bboxes_cur) < 2:
                continue

            # Saves in the BGR format
            cv2.imwrite(img_new_path, new_img[..., ::-1])
            counter += 1
            progress_bar.update(1)
            bboxes.append(bboxes_cur)
            cat_ids.append(cat_ids_cur)
            colors.append(colors_cur)
            del bboxes_cur, cat_ids_cur, colors_cur

        progress_bar.close()

        file = os.path.join(result_imgs_root_fp, f'{subset}_bboxes.pkl')
        write_pkl_unsafe(file, bboxes)
        file = os.path.join(result_imgs_root_fp, f'{subset}_cat_ids.pkl')
        write_pkl_unsafe(file, cat_ids)
        file = os.path.join(result_imgs_root_fp, f'{subset}_colors.pkl')
        write_pkl_unsafe(file, colors)

    print('Created dataset successfully')


def get_new_shape(h, w, target_size=800, max_size=1333):
    old_shape = np.array([h, w])
    new_shape = np.array([h, w])
    index = np.argmax(old_shape)
    aspect_ratio = old_shape[index] / old_shape[1 - index]
    new_shape[1 - index] = target_size
    new_shape[index] = target_size * aspect_ratio
    if new_shape[index] > max_size:
        # print('Nwe shape exceeds limits', new_shape)
        new_shape[index] = max_size
        new_shape[1 - index] = int(max_size / aspect_ratio)

    # print('Old shape', old_shape)
    # print('New shape', new_shape)
    aspect_ratio_old = old_shape[0] / float(old_shape[1])
    aspect_ratio_new = new_shape[0] / float(new_shape[1])
    assert np.abs(aspect_ratio_old - aspect_ratio_new) <= 0.015, f'Old {aspect_ratio_old} New {aspect_ratio_new}'
    return new_shape


if __name__ == '__main__':
    print(get_palette_np())
