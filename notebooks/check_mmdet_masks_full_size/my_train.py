# Copyright (c) OpenMMLab. All rights reserved.
from mmdetection.tools.train import main

if __name__ == '__main__':
    import sys

    args = ['my_mask_rcnn_r50_caffe_c4_1x_coco.py']
    old_sys_argv = sys.argv
    sys.argv = [old_sys_argv[0]] + args
    main()
