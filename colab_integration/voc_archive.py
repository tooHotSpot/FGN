import os
import shutil
from tqdm import tqdm
from typing import List

from cp_utils.cp_dir_file_ops import \
    check_file_if_exists, \
    check_dir_if_exists, \
    create_empty_dir_safe, \
    define_env

if define_env() == 'PC':
    ds_root_path_old = 'C:/Users/Art/PycharmProjects/Datasets/VOC2012'
elif define_env() == 'SERVER':
    ds_root_path_old = '/home/neo/Datasets/VOC2012'
elif define_env() == 'COLAB':
    ds_root_path = '/content/VOC2012'
else:
    raise NotImplementedError('Paths are not specified')

imgs_set_dir_fp = os.path.join(ds_root_path_old, 'ImageSets', 'Segmentation')
assert check_dir_if_exists(imgs_set_dir_fp)

img_set = 'trainval'
file_fp = os.path.join(imgs_set_dir_fp, f'{img_set}.txt')
assert check_file_if_exists(file_fp)
with open(file_fp, mode='r') as f:
    imgs_ids = f.read().splitlines()

dirs: List[str] = ['Annotations', 'JPEGImages', 'SegmentationObject', 'SegmentationClass']
exts: List[str] = ['.xml', '.jpg', '.png', '.png']

for dir_sp in dirs:
    dir_fp = os.path.join(ds_root_path_old, dir_sp)
    assert check_dir_if_exists(dir_fp)

print('Starting to copy')
# SSIS Semantic segmentation and instance segmentation
ds_root_path_new = ds_root_path_old + 'SSIS'
create_empty_dir_safe(ds_root_path_new)

for dir_sp in dirs:
    dir_fp = os.path.join(ds_root_path_new, dir_sp)
    if check_dir_if_exists(dir_fp):
        print('Found already created DIR', dir_fp)
    create_empty_dir_safe(dir_fp)
    print('Created DIR', dir_fp)

dir_fp_old = os.path.join(ds_root_path_old, 'ImageSets')
dir_fp_new = os.path.join(ds_root_path_new, 'ImageSets')

if check_dir_if_exists(dir_fp_new):
    print('Found already created DIR', dir_fp_new)
    assert len(os.listdir(dir_fp_new)) == 4
else:
    shutil.copytree(dir_fp_old, dir_fp_old)

for img_id in tqdm(imgs_ids):
    for dir_, ext_ in zip(dirs, exts):
        file_sp = dir_ + '/' + img_id + ext_
        file_fp_old = os.path.join(ds_root_path_old, file_sp)
        file_fp_new = os.path.join(ds_root_path_new, file_sp)

        if not check_file_if_exists(file_fp_new):
            shutil.copyfile(file_fp_old, file_fp_new)

print('Finished to copy')
