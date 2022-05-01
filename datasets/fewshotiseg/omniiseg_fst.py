import numpy as np

from datasets.omniiseg.omniiseg_ds import OMNIISEG
from datasets.fewshotiseg.mnistiseg_fst import MNISTFewShotISEG


class OMNIFewShotISEG(MNISTFewShotISEG):
    # Class-specific params
    root = '../../datasets/fewshotiseg/resources/omniiseg_fst'
    inner_ds_cl = OMNIISEG

    def __init__(self, config: dict):
        super(OMNIFewShotISEG, self).__init__(config)

    def select_cats(self):
        # There are 26 letters from 0 to 25, choose 6 as novel and 20 as base
        # Chosen manually
        self.cats_total_amount = 26
        self.cat_names_to_ids = {chr(i): (i - 65) for i in range(65, 91)}
        self.cat_ids_to_names = {(i - 65): chr(i) for i in range(65, 91)}
        if self.setup == 'OMNIISEG2OMNIISEG':
            phrase = 'SPUTNIK'
            letters = list(set([self.cat_names_to_ids[c] for c in phrase]))
            self.cats_novel = np.array(sorted(letters), dtype=np.int32)
        elif self.setup == 'OMNIISEG2MNISTISEG':
            self.cats_novel = np.array([], dtype=np.int32)
        elif self.setup == 'MNISTISEG2OMNIISEG':
            self.cats_novel = np.arange(self.cats_total_amount).astype(np.int32)
        else:
            print(f'Incorrect setting {self.setup}')
            raise ValueError


if __name__ == '__main__':
    cfg = dict(
        ds_base_='OMNIISEG',
        ds_base__subset='train',
        ds_novel='OMNIISEG',
        ds_novel_subset='val',
        sampling_origin_ds='OMNIISEG',
        sampling_origin_ds_subset='train',
        sampling_cats='base_',
        first_parents__only=0,
        first_children_only=0,
        sampling_scenario='parents',
        repeats=0,
        finetune='Ignore'
    )

    ds = OMNIFewShotISEG(config=cfg)
    print(ds.root)
    print(ds.imgs_dir_fp)
    print(ds.mean)
    print(ds.std)
    ds.visualize(n_imgs=10)
