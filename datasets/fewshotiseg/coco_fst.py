import numpy as np
from imgaug import augmenters as iaa

from cp_utils.cp_dir_file_ops import define_env
from datasets.coco.coco_ds import COCODS
from datasets.fewshotiseg.mnistiseg_fst import MNISTFewShotISEG


class COCOFewShot(MNISTFewShotISEG):
    # Class-specific params
    root = '../../datasets/fewshotiseg/resources/coco_ds_fst'
    inner_ds: COCODS
    inner_ds_cl = COCODS
    # More common params
    spp_img_size = 256
    augs_qry: iaa.Sequential = COCODS.augs_seq
    augs_spp: iaa.Sequential = COCODS.augs_seq

    def __init__(self, config: dict):
        super(COCOFewShot, self).__init__(config)

    def select_cats(self):
        voc_cats = {"person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane",
                    "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair",
                    "dining table", "potted plant", "sofa", "tvmonitor"}
        extra = {'aeroplane': 'airplane',
                 'sofa': 'couch',
                 'tvmonitor': 'tv',
                 'motorbike': 'motorcycle'}

        # Totally 80 categories, 20 in VOC are for test
        self.cats_total_amount = len(self.inner_ds.new_cats_ids_to_cats_names)
        self.cat_names_to_ids = self.inner_ds.cats_names_to_new_cats_ids
        self.cat_ids_to_names = self.inner_ds.new_cats_ids_to_cats_names
        # These are not required in general but left here for future purpose
        # self.new_cats_ids_to_original = self.inner_ds.new_cats_ids_to_original
        # self.original_to_new_cats_ids = self.inner_ds.original_to_new_cats_ids
        # self.inner_ds is deleted after this method
        if self.setup == 'COCO2VOC':
            # Chosen to make recognition harder since 1 ~ 7 and 3 ~ 5 in MNIST
            cats_novel = []
            for name in voc_cats:
                # Some cats in VOC are named differently
                if name not in self.cat_names_to_ids:
                    name = extra[name]
                new_id = self.cat_names_to_ids[name]
                cats_novel.append(new_id)

            if len(cats_novel) != 20:
                print('In COCO2VOC settings found only', len(cats_novel), 'cats instead of 20 required')
            self.cats_novel = np.array(sorted(cats_novel), dtype=np.int32)
        else:
            print(f'Incorrect setting {self.setup}')
            raise ValueError


if __name__ == '__main__':
    cfg = dict(
        verbose=False,
        ds_base_='COCO',
        ds_base__subset='train',
        ds_novel='VOC',
        ds_novel_subset='val',
        sampling_origin_ds='COCO',
        sampling_origin_ds_subset='train',
        sampling_cats='base_',
        first_parents__only=0,
        first_children_only=0,
        augment_qry=False,
        augment_spp=False,
        # parents, children
        sampling_scenario='children',
        repeats=1,
        shuffle=False,
        qry_cats_choice_remove=False,
        qry_cats_choice_random=False,
        qry_cats_order_shuffle=True,
        spp_random=True,
        delete_qry_insts_in_spp_insts_on_train=True,
        finetune='Ignore',
        spp_fill_ratio=0.9,
        spp_crop_square=True
    )

    ds = COCOFewShot(config=cfg)
    print(ds.root)
    ds.visualize(n_imgs=100, choose_random=True)
