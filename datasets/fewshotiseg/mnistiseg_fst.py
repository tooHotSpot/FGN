import gc
import os
import numpy as np
import imgaug.augmenters as iaa

from datasets.mnistiseg.mnistiseg_ds import MNISTISEG
from datasets.fewshotiseg.base_fst import BaseFewShotISEG


class MNISTFewShotISEG(BaseFewShotISEG):
    # Class-specific params
    root = '../../datasets/fewshotiseg/resources/mnistiseg_fst'
    inner_ds: MNISTISEG
    inner_ds_cl = MNISTISEG
    # More common params
    spp_img_size = 128
    augs_qry: iaa.Sequential = MNISTISEG.augs_seq
    augs_spp: iaa.Sequential = MNISTISEG.augs_seq

    def __init__(self, config: dict):
        super(MNISTFewShotISEG, self).__init__(config)

        read_data = not os.path.exists(self.databag_fp)
        # Reading whole data, feeding no `first_only` or `offset` flags
        inner_ds = self.inner_ds_cl(imgs_set=self.sampling_origin_ds_subset, read_data=read_data)

        # Using a second base class would be a greater idea but
        # I could not handle various errors encountered during runtime
        for attr in ('target_size',
                     'max_size',
                     'mean',
                     'std',
                     'imgs_dir_fp',
                     'transforms',
                     'denormalize',
                     'get_isegmap'):
            setattr(self, attr, getattr(inner_ds, attr))
        if read_data:
            self.imgs_sps = inner_ds.imgs_sps
            self.bboxes = inner_ds.bboxes
            self.cat_ids = inner_ds.cat_ids
            self.info_isegmaps = inner_ds.info_isegmaps

        # The inner_ds is required for COCOFewShot.
        # This may be changed to something more wise.
        self.inner_ds = inner_ds
        self.select_cats()
        del self.inner_ds, inner_ds
        gc.collect()

        print('Novel cats')
        print([self.cat_ids_to_names[c] for c in self.cats_novel])

        self.load_dataset()

    def select_cats(self):
        # There are 10 numbers from 0 to 9, choose 4 as novel and 6 as base
        # Chosen manually
        self.cats_total_amount = 10
        self.cat_names_to_ids = {f'{i}': i for i in range(10)}
        self.cat_ids_to_names = {i: f'{i}' for i in range(10)}
        if self.setup == 'MNISTISEG2MNISTISEG':
            # Chosen to make recognition harder since 1 ~ 7 and 3 ~ 5 in MNIST
            self.cats_novel = np.array([1, 3, 5, 7], dtype=np.int32)
        elif self.setup == 'MNISTISEG2OMNIISEG':
            self.cats_novel = np.array([], dtype=np.int32)
        elif self.setup == 'OMNIISEG2MNISTISEG':
            self.cats_novel = np.arange(self.cats_total_amount).astype(np.int32)
        else:
            print(f'Incorrect setting {self.setup}')
            raise ValueError


if __name__ == '__main__':
    cfg = dict(
        ds_base_='MNISTISEG',
        ds_base__subset='train',
        ds_novel='MNISTISEG',
        ds_novel_subset='val',
        sampling_origin_ds='MNISTISEG',
        sampling_origin_ds_subset='train',
        sampling_cats='base_',
        first_parents__only=0,
        first_children_only=0,
        sampling_scenario='parents',
        repeats=0,
        finetune='Ignore'
    )

    ds = MNISTFewShotISEG(config=cfg)
    print(ds.root)
    print(ds.imgs_dir_fp)
    print(ds.mean)
    print(ds.std)
    ds.visualize(n_imgs=10)
