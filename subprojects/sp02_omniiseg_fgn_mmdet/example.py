from mmcv import imread
from mmdet.apis import init_detector, inference_detector

config_file = '../../mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_c4_1x_coco.py'
device = 'cpu'
# init a detector
model = init_detector(config_file, device=device)
# inference the demo image
img_path = '../../mmdetection/demo/demonew.jpg'
img = imread(img_path)

result = inference_detector(model, img)
print('Finish')
