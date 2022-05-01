# Labels to colors and backward mappings
# Source: https://github.com/chainer/chainercv/blob/master/chainercv/datasets/voc/voc_utils.py

voc_labels_colors = {
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'bird': (128, 128, 0),
    'boat': (0, 0, 128),
    'bottle': (128, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'cat': (64, 0, 0),
    'chair': (192, 0, 0),
    'cow': (64, 128, 0),
    'diningtable': (192, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'motorbike': (64, 128, 128),
    'person': (192, 128, 128),
    'pottedplant': (0, 64, 0),
    'sheep': (128, 64, 0),
    'sofa': (0, 192, 0),
    'train': (128, 192, 0),
    'tvmonitor': (0, 64, 128)
}

voc_colors_labels = {
    (128, 0, 0): 'aeroplane',
    (0, 128, 0): 'bicycle',
    (128, 128, 0): 'bird',
    (0, 0, 128): 'boat',
    (128, 0, 128): 'bottle',
    (0, 128, 128): 'bus',
    (128, 128, 128): 'car',
    (64, 0, 0): 'cat',
    (192, 0, 0): 'chair',
    (64, 128, 0): 'cow',
    (192, 128, 0): 'diningtable',
    (64, 0, 128): 'dog',
    (192, 0, 128): 'horse',
    (64, 128, 128): 'motorbike',
    (192, 128, 128): 'person',
    (0, 64, 0): 'pottedplant',
    (128, 64, 0): 'sheep',
    (0, 192, 0): 'sofa',
    (128, 192, 0): 'train',
    (0, 64, 128): 'tvmonitor',
}

voc_background_color = (0, 0, 0)
voc_ignore_label_color = (224, 224, 192)

voc_labels = sorted(list(voc_labels_colors))
voc_labels_codes = {label: i for i, label in enumerate(voc_labels)}
voc_codes_labels = {i: label for i, label in enumerate(voc_labels)}

