import os
import sys
import shutil
import gc
from tqdm import tqdm
from time import time

import cv2
import PIL
from PIL import Image
from skimage import io as skimageio

import numpy as np
from io import BytesIO

from cp_utils.cp_time import datetime_diff, datetime_diff_ms, datetime_now
from cp_utils.cp_dir_file_ops import define_env, check_dir_if_exists


class Subsets:
    train = 'train'
    val = 'val'


subset = Subsets.val

coco_ds_path = None
if define_env() == 'PC':
    coco_ds_path = f'D:/Datasets/COCO/{subset}2017'
elif define_env() == 'SERVER':
    coco_ds_path = f'/home/neo/Datasets/COCO/{subset}2017'
elif define_env() == 'COLAB':
    coco_ds_path = f'/content/COCO/{subset}2017'
else:
    raise NotImplementedError

print('COCO DS path:')
print(coco_ds_path)
print('Path exists')
answer = check_dir_if_exists(coco_ds_path)
print(answer)
assert answer

imgs_list_us = os.listdir(coco_ds_path)
print('Total images', len(imgs_list_us))
t1 = datetime_now()
imgs_list = sorted(imgs_list_us)
print('Sorted in ', datetime_diff_ms(t1))

limit = -1
if limit == -1:
    limit = len(imgs_list)
else:
    limit = min(len(imgs_list), limit)

t1 = datetime_now()
total = 0
for i, img_sp in tqdm(enumerate(imgs_list[:limit]), total=limit, ncols=50):
    img_fp = os.path.join(coco_ds_path, img_sp)
    try:
        if i == 0:
            img = cv2.imread(img_fp, flags=cv2.IMREAD_COLOR)
            print('')
            print('IMG 0 Shape', img.shape)

        # Method 1 (Results in 0 corrupt images in the COCO-val)
        # with open(img_fp, 'rb') as f:
        #     chars = f.read()
        # check_chars = chars[-2:]
        # if check_chars != b'\xff\xd9':
        #     raise BaseException

        # Method 2 (Results in all 5000 images in COCO0val as corrupt)
        # buff = BytesIO()
        # buff.write(chars)
        # buff.seek(0)
        # temp_img = np.array(PIL.Image.open(buff), dtype=np.uint8)
        # img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)

        # Method 3 (Results in 0 corrupt images in the COCO-val)
        # skimageio.imread(img_fp)

        # Check with means of PIL and other
        statfile = os.stat(img_fp)
        filesize = statfile.st_size
        if filesize == 0:
            print('0 SIZE:', img_sp)
            continue

        im = Image.open(img_fp)
        im.verify()
        im.close()
        im = Image.open(img_fp)
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        im.close()
    except:
        print('Corrupt', img_fp)
        total += 1

print('Total corrupt', total)
print('Loop finished in ', datetime_diff(t1))
