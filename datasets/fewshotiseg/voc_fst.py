from imgaug import augmenters as iaa

from datasets.voc.voc_ds import VOCDS
from datasets.fewshotiseg.coco_fst import COCOFewShot


class VOCFewShot(COCOFewShot):
    # Class-specific params
    root = '../../datasets/fewshotiseg/resources/voc_ds_fst'
    inner_ds: VOCDS
    inner_ds_cl = VOCDS
    # More common params
    spp_img_size = 256

    def __init__(self, config: dict):
        super(VOCFewShot, self).__init__(config)

        if self.finetune == 'Use':
            self.augs_qry: iaa.Sequential = iaa.Sequential([])
            self.augs_spp: iaa.Sequential = iaa.Sequential([])

            print('Augmentations for the VOCFewShot are deleted for FT=Use option')
            print('self.augs_qry', self.augs_qry)
            print('self.augs_spp', self.augs_spp)


if __name__ == '__main__':
    cfg = dict(
        ds_base_='COCO',
        ds_base__subset='train',
        ds_novel='VOC',
        ds_novel_subset='val',
        sampling_origin_ds='VOC',
        sampling_origin_ds_subset='trainval',
        sampling_cats='novel',
        first_parents__only=0,
        first_children_only=0,
        sampling_scenario='children',
        repeats=1,
        finetune='Use'
    )

    ds = VOCFewShot(cfg)
    print(ds.root)
    ds.visualize()
