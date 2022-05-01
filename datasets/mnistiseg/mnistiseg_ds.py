import os
from tqdm import tqdm
from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from torch import Tensor

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import imgaug.augmenters as iaa

from datasets.fewshotiseg.base_fst import BaseFewShotISEG
from cp_utils.cp_dir_file_ops import create_empty_dir_unsafe, read_pkl, write_json_unsafe
from cp_utils.create_img_from_chars import get_char_mask_by_color


class MNISTISEG(Dataset):
    img_size = 512
    target_size = 480
    max_size = 480
    imgs_set_possible = ('train', 'val', 'test')

    imgs_set: str = ''
    imgs_dir_fp: str = ''
    imgs_sps: list = []
    bboxes: Union[list, ndarray] = []
    cat_ids: Union[list, ndarray] = []
    info_isegmaps: list = []
    len: int = 0
    first_only: int = 0

    # Counted OMNIISEG with a double precision
    mean = np.array([0.953, 0.952, 0.953], dtype=np.float32)
    std = np.array([0.168, 0.168, 0.166], dtype=np.float32)
    root = '../../datasets/mnistiseg/resources'
    samples_dir = 'mnistiseg_samples'

    # Augmentation sequence for a synthetic query image.
    augs_seq = iaa.Sequential([
        iaa.SomeOf(1, [
            iaa.Affine(translate_px=(-15, 15), mode='edge'),
            iaa.Affine(scale=(0.8, 1.2), mode='edge'),
            iaa.Affine(rotate=(-15, 15), mode='edge'),
            iaa.Affine(shear=(-5, 5), mode='edge'),
        ]),
        iaa.SomeOf(1, [
            iaa.AdditiveGaussianNoise(loc=0, scale=1),
            iaa.ImpulseNoise(),
            iaa.GaussianBlur(),
            iaa.AddToHue(value=(-50, 50))
        ]),
    ])

    def __init__(self, imgs_set='train', augment=False, first_only=0, offset=0, read_data=True):
        super(MNISTISEG, self).__init__()
        assert imgs_set in self.imgs_set_possible
        self.imgs_set = imgs_set
        self.imgs_dir_fp = os.path.join(self.root, imgs_set)
        self.augment = augment
        self.offset = offset
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        if read_data:
            self.read_data()

            # Check lengths of all data arrays after loading
            print('Total amount of images       ', len(self.imgs_sps))
            print('Total amount of bboxes       ', len(self.bboxes))
            print('Total amount of cat_ids      ', len(self.cat_ids))
            print('Total amount of info_isegmaps', len(self.info_isegmaps))
            assert len(self.imgs_sps) == len(self.bboxes) == \
                   len(self.cat_ids) == len(self.info_isegmaps)
            print('All lengths are same')

            if 0 < first_only <= len(self.imgs_sps):
                self.imgs_sps = self.imgs_sps[:first_only]
                self.bboxes = self.bboxes[:first_only]
                self.cat_ids = self.cat_ids[:first_only]
                self.info_isegmaps = self.info_isegmaps[:first_only]
                print('Reduced amount of samples to first', first_only)
            elif first_only != 0:
                print('Strange value of the <first_only> arg', first_only)

            self.len = len(self.imgs_sps)
            self.first_only = len(self.imgs_sps)
        return

    def read_data(self):
        self.imgs_sps = sorted(os.listdir(self.imgs_dir_fp))
        # Reading all the data from several files
        self.bboxes = read_pkl(os.path.join(self.root, f'{self.imgs_set}_bboxes.pkl'))
        self.cat_ids = read_pkl(os.path.join(self.root, f'{self.imgs_set}_cat_ids.pkl'))
        self.info_isegmaps = read_pkl(os.path.join(self.root, f'{self.imgs_set}_colors.pkl'))

    def __len__(self):
        return len(self.imgs_sps)

    @staticmethod
    def get_isegmap(img: ndarray, bbox: Union[list, ndarray], info: Union[list, ndarray]) -> ndarray:
        assert len(info) == 3
        color = info
        return get_char_mask_by_color(img, bbox, color)

    def __getitem__(self, idx):
        idx = (idx + self.offset) % self.len
        # Read an image
        t_path = os.path.join(self.imgs_dir_fp, self.imgs_sps[idx])
        img = cv2.imread(t_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Read a metadata
        cat_ids = np.array(self.cat_ids[idx], dtype=np.int32)
        bboxes = np.array(self.bboxes[idx], dtype=np.int32)
        info = self.info_isegmaps[idx]
        isegmaps = []
        for i in range(len(bboxes)):
            arr = self.get_isegmap(img, bboxes[i], info[i])
            isegmaps.append(arr)
        isegmaps = np.array(isegmaps, dtype=bool)

        if self.augment:
            img, bboxes, isegmaps = \
                BaseFewShotISEG.augment_with_imgaug(self.augs_seq, img, bboxes, isegmaps)

        if self.transforms:
            img = self.transforms(img)

        sample = {
            'img': img,
            'cat_ids': torch.from_numpy(cat_ids).long(),
            'bboxes': torch.from_numpy(bboxes.astype(np.float32)),
            'isegmaps': torch.from_numpy(isegmaps),
            'img_id': Tensor([idx])
        }
        return sample

    def denormalize(self, img: Union[ndarray, Tensor]) -> ndarray:
        assert len(img.shape) == 3, f'Invalid shape (only 3 are OK): {img.shape}'
        assert img.shape[0] == 3, f'Invalid channels num (only 3 are OK): {img.shape[0]}'
        if isinstance(img, Tensor):
            if img.device.type != 'cpu':
                img = img.detach().cpu()
            img = img.numpy()
        a_np = img.transpose((1, 2, 0))
        a_denormalize = (a_np * self.std) + self.mean
        a_uint8 = (a_denormalize * 255).astype(np.uint8)
        return a_uint8

    def show(self, img: ndarray) -> None:
        plt.imshow(img, cmap='gray')
        plt.show()
        return

    def count_mean_std(self, n_imgs=1000):
        """
        Use decomposition formulas in order to count mean and std of big set of
         images, which does not require a huge amount of memory.

        :param n_imgs: amount of images to count statistics on
        :return:
        """
        self.transforms = None
        samples = np.random.choice(self.len, size=n_imgs, replace=False)
        img_mean = np.zeros((self.img_size, self.img_size, 3), dtype=np.float64)
        img_var = np.zeros((self.img_size, self.img_size, 3), dtype=np.float64)

        # Optimization with a PyTorch Dataloader is available
        for i, sample_idx in tqdm(enumerate(samples)):
            item = self.__getitem__(idx=sample_idx)
            # Scale to [0, 1]
            img = item['img'].astype(np.float64) / 255
            img_mean += img
            img_var += img ** 2

        means = img_mean.sum(axis=(0, 1)) / (self.img_size * self.img_size * n_imgs)
        variances = img_var.sum(axis=(0, 1)) / (self.img_size * self.img_size * n_imgs) - means ** 2
        stds = np.sqrt(variances)

        params = {
            'mean': means.astype(np.float32).tolist(),
            'std': stds.astype(np.float32).tolist(),
            'n_imgs': n_imgs,
            'imgs_size': self.img_size
        }
        write_json_unsafe(f'Params{self.__class__.__name__}.json', params)
        return

    def visualize(self, with_isegmaps=False, seed=8, n_imgs=20, choose_random=False, action='save', vis_dir=None):
        assert action in ('save', 'show')

        if vis_dir is None:
            vis_dir = os.path.join(self.root, 'visualize_examples')
        if action == 'save':
            print('Saving visualize examples to the', vis_dir)
        create_empty_dir_unsafe(vis_dir)

        samples = []
        if choose_random and n_imgs <= self.len:
            np.random.seed(seed)
            samples = np.random.choice(self.len, size=n_imgs, replace=False)
        elif n_imgs > 0:
            samples = np.arange(n_imgs)
        else:
            print('Could not sample any image cause the dataset is empty')
            return

        transforms_old = self.transforms
        self.transforms = None
        for i, sample_idx in tqdm(enumerate(samples)):
            item = self.__getitem__(idx=sample_idx)
            img = item['img']
            # Convert from tensors to ndarrays
            cat_ids = item['cat_ids'].numpy()
            bboxes = item['bboxes'].numpy()
            bbs = BaseFewShotISEG.get_bboxes_on_img_from_yxyx(img=img, bboxes=bboxes, cat_ids=cat_ids)
            img = bbs.draw_on_image(img, size=2, color=[0, 255, 0])
            if with_isegmaps:
                isegmaps = item['isegmaps'].numpy()
                isegmaps = BaseFewShotISEG.get_isegmaps_on_img_multiple(img=img, isegmaps=isegmaps)
                for j in range(len(isegmaps)):
                    img = isegmaps[j].draw_on_image(img, alpha=0.8)[0]

            if action == 'save':
                path = os.path.join(vis_dir, f'Image {sample_idx:05}.png')
                plt.imsave(path, img, cmap='gray')
                plt.close('all')
            else:
                plt.imshow(img)
                plt.show()

        self.transforms = transforms_old
        return


def example(_ds):
    dloader = DataLoader(_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    cv2.ocl.setUseOpenCL(False)

    for i, samples in enumerate(dloader):
        img = samples['img']
        print('images', img.shape)
        print('bboxes', samples['bboxes'])
        print('cat_ids', samples['cat_ids'])
        print('img_id', samples['img_id'])
        isegmaps = samples['isegmaps']
        print('isegmaps', isegmaps.shape)
        break
    print('Finished')


if __name__ == '__main__':
    ds = MNISTISEG(augment=True, imgs_set='train')
    print(ds.mean)
    print(ds.std)
    example(ds)
    ds.visualize(n_imgs=10, with_isegmaps=False, action='save')
