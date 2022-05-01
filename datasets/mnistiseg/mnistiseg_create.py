import os

from cp_utils.create_img_from_chars import create_ds
from cp_utils.cp_dir_file_ops import check_dir_if_exists


def create_mnistiseg_ds():
    DEBUG = False
    origin_imgs_root_fp = '../../yymnist/mnist'
    new_imgs_root_fp = '../../datasets/mnistiseg/resources'

    data_train_path = os.path.join(origin_imgs_root_fp, 'train')
    assert check_dir_if_exists(data_train_path)
    data_test_path = os.path.join(origin_imgs_root_fp, 'test')
    assert check_dir_if_exists(data_test_path)
    # Train and val are created from train images
    # Test is created from test images to be father from the train distribution
    data_train_imgs_paths = [os.path.join(data_train_path, e)
                             for e in os.listdir(data_train_path)]
    data_test_imgs_paths = [os.path.join(data_test_path, e)
                            for e in os.listdir(data_test_path)]

    root = '.'
    resources = os.path.join(root, 'resources')
    if not os.path.exists(root):
        os.mkdir(resources)

    # 5000 train, 500 val, no test for this simple dataset
    new_subsets_quantities = {
        'train': 4000,
        'val': 500,
        'test': 500
    }
    if DEBUG:
        for subset in new_subsets_quantities:
            new_subsets_quantities[subset] = 50
    new_subset_origin_imgs_paths = {
        'train': data_train_imgs_paths,
        'val': data_train_imgs_paths,
        'test': data_test_imgs_paths,
    }
    # Using large, medium, small in the alphabetical order
    sizes_max_amount = {
        'small': 2,
        'medium': 2,
        'large': 2
    }
    sizes_min_max_ratios = {
        'small': [4, 8],
        'medium': [8, 12],
        'large': [12, 15]
    }
    new_imgs_size_mnistiseg = 512
    create_ds(new_subset_origin_imgs_paths, new_subsets_quantities,
              sizes_max_amount, sizes_min_max_ratios,
              new_imgs_root_fp, img_new_size=new_imgs_size_mnistiseg)


if __name__ == '__main__':
    create_mnistiseg_ds()
