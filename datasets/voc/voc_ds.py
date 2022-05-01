import os
import numpy as np

from cp_utils.cp_dir_file_ops import define_env, check_file_if_exists
from cp_utils.cp_dir_file_ops import read_pkl, write_pkl_safe

from datasets.coco.coco_ds import COCODS
from datasets.voc.voc_ds_parse import VOCDSParse

if define_env() == 'PC':
    ds_root_path = 'C:/Users/Art/PycharmProjects/Datasets/VOC2012'
elif define_env() == 'SERVER':
    ds_root_path = '/home/neo/Datasets/VOC2012'
elif define_env() == 'COLAB':
    ds_root_path = '/content/VOC2012'
else:
    raise NotImplementedError('Paths are not specified')


class VOCDS(COCODS):
    target_size = 512
    max_size = 512
    # There are train, val and trainval, but only train and val are used
    imgs_set_possible = ('train', 'val', 'trainval')

    # ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    root = '../../datasets/voc/resources'
    samples_dir = 'voc_samples'

    # Code development flags
    is_list = 0
    has_counts = 0
    simple = 0

    def __init__(self, **kwargs):
        super(VOCDS, self).__init__(**kwargs)
        self.imgs_dir_fp = os.path.join(ds_root_path, 'JPEGImages')

    def read_data(self):
        name = str.upper(self.imgs_set)
        parsed_data_coco_new_path = os.path.join(self.root, f'PARSED_DATA_{name}_COCO_NEW.pkl')
        del name

        if not check_file_if_exists(parsed_data_coco_new_path):
            ds = VOCDSParse(img_set=self.imgs_set)
            exc = ds.get_excluded()
            img_sps, bboxes, cat_ids, info_isegmaps = ds.get_results(exc)

            for i in range(len(cat_ids)):
                cat_ids[i] = self.original_to_new_cats_ids[cat_ids[i]]

            data_new = [img_sps, bboxes, cat_ids, info_isegmaps]
            write_pkl_safe(parsed_data_coco_new_path, data_new)
            print('Wrote data with NEW COCO CAT_IDS to', parsed_data_coco_new_path)
        else:
            data_new = read_pkl(parsed_data_coco_new_path)
            img_sps, bboxes, cat_ids, info_isegmaps = data_new
            print('Read data with NEW COCO CAT_IDS to', parsed_data_coco_new_path)

        # Change
        self.imgs_sps = img_sps
        self.bboxes = bboxes
        self.cat_ids = cat_ids
        self.info_isegmaps = info_isegmaps

        print('Finished with reading VOC data')
        self.len = len(self.imgs_sps)
        self.first_only = len(self.imgs_sps)
        return


if __name__ == '__main__':
    vds = VOCDS(imgs_set='trainval', augment=False)
    vds.visualize(n_imgs=10, with_isegmaps=True, action='save')
