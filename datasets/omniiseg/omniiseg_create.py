# This code is almost a copy of a corresponding method for a MNISTISEG.
# Code is repeated because of different images paths accumulation loop
# and some numerical settings.
import os
from cp_utils.cp_dir_file_ops import check_dir_if_exists
from cp_utils.create_img_from_chars import create_ds


def create_omniiseg_ds():
    DEBUG = False
    origin_imgs_root_fp = '../../omniglot/python/images_background/Latin'
    new_imgs_root_fp = '../../datasets/omniiseg/resources'

    assert check_dir_if_exists(origin_imgs_root_fp), f'Not a dir: {origin_imgs_root_fp}'
    assert check_dir_if_exists(new_imgs_root_fp), f'Not a dir: {new_imgs_root_fp}'

    data_train_imgs_paths = []
    data_val_imgs_paths = []
    data_test_imgs_paths = []

    for char_dir_sp in sorted(os.listdir(origin_imgs_root_fp)):
        char_dir_fp = os.path.join(origin_imgs_root_fp, char_dir_sp)
        chars_imgs_sps = sorted(os.listdir(char_dir_fp))
        char_imgs_fps = [os.path.join(char_dir_fp, img_sp)
                         for img_sp in chars_imgs_sps]
        # 20 imgs are separated into 10 training, 5 val, 5 test
        data_train_imgs_paths.extend(char_imgs_fps[:10])
        data_val_imgs_paths.extend(char_imgs_fps[10:15])
        data_test_imgs_paths.extend(char_imgs_fps[15:])

    root = '.'
    resources = os.path.join(root, 'resources')
    if not os.path.exists(root):
        os.mkdir(resources)

    # 8000 train, 1000 val
    new_subsets_quantities = {
        'train': 8000,
        'val': 1000,
        'test': 1000
    }
    if DEBUG:
        for subset in new_subsets_quantities:
            new_subsets_quantities[subset] = 50
    new_subset_origin_imgs_paths = {
        'train': data_train_imgs_paths,
        'val': data_val_imgs_paths,
        'test': data_test_imgs_paths,
    }
    # Using large, medium, small in the alphabetical order
    sizes_max_amount = {
        'small': 2,
        'medium': 2,
        'large': 2
    }
    sizes_min_max_ratios = {
        'small': [1, 2.5],
        'medium': [2.5, 4.0],
        'large': [4.0, 5]
    }
    new_imgs_size_omniiseg = 512
    create_ds(new_subset_origin_imgs_paths, new_subsets_quantities,
              sizes_max_amount, sizes_min_max_ratios,
              new_imgs_root_fp, img_new_size=new_imgs_size_omniiseg,
              cat_id_in_upper_dir=True)


if __name__ == '__main__':
    create_omniiseg_ds()
