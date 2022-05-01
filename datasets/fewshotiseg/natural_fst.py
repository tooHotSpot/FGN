import numpy as np
from numpy import ndarray

from imgaug import augmenters as iaa

from torchvision import transforms


class NaturalDataset:
    target_size = 800
    max_size = 1333
    spp_img_size = 512

    # Imagenet values
    mean: ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std: ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    augs_seq = iaa.Sequential(children=[
        iaa.Sometimes(0.5, iaa.SomeOf(n=1, children=[
            iaa.AdditiveGaussianNoise(loc=0, scale=1),
            iaa.ImpulseNoise(),
            iaa.GaussianBlur(),
            iaa.AddToHue(value=(-50, 50)),
            iaa.AddToSaturation(),
            iaa.AddToBrightness(),
            iaa.ChannelShuffle(),
        ])),
        iaa.Sometimes(0.5, iaa.SomeOf(n=1, children=[
            iaa.Fliplr(),
            iaa.Affine(rotate=10, mode='symmetric'),
            iaa.Affine(scale=(0.8, 1.2)),
            iaa.Affine(shear=(-10, 10), mode='symmetric'),
            iaa.Affine(translate_px=(-20, 20))
        ]))
    ])

    augs_spp = iaa.Sequential([
        iaa.SomeOf(n=1, children=[
            iaa.ChannelShuffle(),
            iaa.GammaContrast(gamma=(0.95, 1.05), per_channel=False),
            iaa.Sharpen(),
        ]),
        iaa.SomeOf(n=1, children=[
            iaa.AdditiveGaussianNoise(),
            iaa.AddToHue(value=(-75, 75)),
            iaa.AddToBrightness(add=(-30, 30)),
            iaa.AddToSaturation(value=(-75, 75)),
        ]),
        iaa.Sometimes(p=0.5, then_list=[iaa.Fliplr()]),
    ])

    def __init__(self):
        super(NaturalDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

