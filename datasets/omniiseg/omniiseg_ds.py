# Since datasets are similar to each other, code is significantly reduced.
import numpy as np

from datasets.mnistiseg.mnistiseg_ds import MNISTISEG, example


class OMNIISEG(MNISTISEG):
    mean = np.array([0.963, 0.964, 0.963], dtype=np.float32)
    std = np.array([0.160, 0.158, 0.159], dtype=np.float32)
    root = '../../datasets/omniiseg/resources'
    samples_dir = 'omniiseg_samples'

    def __init__(self, **kwargs):
        super(OMNIISEG, self).__init__(**kwargs)


if __name__ == '__main__':
    ds = OMNIISEG(augment=True, imgs_set='val')
    print(ds.mean)
    print(ds.std)
    example(ds)
    ds.visualize(n_imgs=10, with_isegmaps=False, action='save')
