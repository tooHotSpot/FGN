import os
import contextlib
import gc
import time
import traceback
import logging
import warnings

from tqdm import tqdm

from typing import List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

from mmcv import Config
# from mmcv.runner import load_checkpoint,
from mmcv.runner import build_optimizer, EpochBasedRunner
from mmcv.runner.hooks.logger import TensorboardLoggerHook

from mmcv.runner.base_module import print_log
from mmcv.runner.base_module import BaseModule

# from mmdet.apis import single_gpu_test
# from mmdet.core import EvalHook
# from mmcv.runner.builder import RUNNERS
from mmcv.runner import EvalHook as BaseEvalHook
from mmdet.utils import get_root_logger
from mmdet.models import build_detector

from cp_utils.cp_time import datetime_log_fancy
from cp_utils.cp_dir_file_ops import define_env, debugger_is_active, give_mem
from cp_utils.cp_dir_file_ops import \
    check_dir_if_exists, \
    create_empty_dir_safe, \
    create_empty_dir_unsafe
from cp_utils.cp_dir_file_ops import write_pkl_safe
from cp_utils.cp_time import datetime_now, datetime_diff, datetime_diff_ms
from datasets.fewshotiseg.base_fst import BaseFewShotISEG
from datasets.fewshotiseg.mnistiseg_fst import MNISTFewShotISEG
from datasets.fewshotiseg.omniiseg_fst import OMNIFewShotISEG
from datasets.fewshotiseg.coco_fst import COCOFewShot
from datasets.fewshotiseg.voc_fst import VOCFewShot
from subprojects.sp02_omniiseg_fgn_mmdet.fgn import AGRPNHead, FGNRoIHead, FGN

from subprojects.sp02_omniiseg_fgn_mmdet.main import \
    collate_fn_new, \
    OptEpochBasedRunner, \
    OptEvalHook, \
    init_ds_class_by_config, \
    eval_hooks_list, \
    main

if __name__ == '__main__':

    '''
    upper = '/home/neo/PycharmProjects/Course1/subprojects/sp02_omniiseg_fgn_mmdet/'
    ch11 = '2022-02-12_13-12-07_Train-1-1-Scratch-MNISTISEG-Train ' \
           'LR 0.01 WD 5e-05 Adam NoLRRescale CosineLRDecayMinLRRatio 0.01 BEST/epoch_10.pth'
    ch31 = '2022-02-13_09-31-27_Train-3-1-Scratch-MNISTISEG-Train ' \
           'LR 0.01 WD 5e-05 Adam Cosine0.01 NoLRRescale BEST/epoch_10.pth'
    ch33 = '2022-02-15_08-24-08_Train-3-3-Scratch-MNISTISEG-Train ' \
           'LR 0.01 WD 5e-05 Adam NoLRRescale CosineLRDecayMinLRRatio 0.01 BEST/epoch_10.pth'
    '''

    upper = '/home/neo/PycharmProjects/Course1/subprojects/sp02_omniiseg_fgn_mmdet/COCO2VOC_DCL'
    ch11 = 'N1_K1_B12_2022-02-24_11-27-34 C2V-DCL LR 0.005 WD 1e-05 Adam STANDARD/epoch_4.pth'
    ch31 = 'N3_K1_B10_2022-02-23_23-11-39 C2V-DCL LR 0.005 WD 1e-05 Adam STANDARD/epoch_4.pth'
    ch33 = 'N3_K3_B8_2022-02-23_22-00-04 C2V-DCL LR 0.005 WD 1e-05 Adam STANDARD/epoch_4.pth'

    for gamma in (0.01, 0.05, 0.1):
        for n_ways, k_shots, checkpoint_sp in ((1, 1, ch11), (3, 1, ch31), (3, 3, ch33)):
            config_file = 'fgn_ft.py'
            cfg = Config.fromfile(config_file)

            # Change values in FGN and values in cfg.datasets (for their check in main())
            # Change gamma values
            FGN.n_ways = n_ways
            FGN.k_shots = k_shots
            cfg._cfg_dict['model']['n_ways'] = n_ways
            cfg._cfg_dict['model']['k_shots'] = k_shots

            # Change here
            # MNIST
            # ds_names = [
            #     'ft_ds_cfg0',
            #     'ft_ds_cfg1',
            #     'eval_ds_cfg0',
            #     'eval_ds_cfg1',
            #     'eval_ds_cfg2',
            #     'eval_ds_cfg3'
            # ]
            # COCO2VOC
            ds_names = [
                'ft_ds_cfg0',
                'ft_ds_cfg1',
                'eval_ds_cfg0'
            ]
            for name in ds_names:
                cfg._cfg_dict[name]['n_ways'] = n_ways
                cfg._cfg_dict[name]['k_shots'] = k_shots

            # Change checkpoint from None
            chkp_cur = cfg.get('checkpoint')
            assert f'N{n_ways}_K{k_shots}' in checkpoint_sp, f'Bad ckpt {checkpoint_sp} for N{n_ways}_K{k_shots}'
            cfg._cfg_dict['checkpoint'] = os.path.join(upper, checkpoint_sp)
            chkp_new = cfg.get('checkpoint')
            print('CHECKPOINT', chkp_new)
            assert os.path.exists(chkp_new), f'Bad path: {chkp_new}'

            cfg._cfg_dict['lr_config']['gamma'] = gamma
            gamma_new = cfg.get('lr_config')['gamma']
            print('GAMMA', gamma_new)
            assert gamma_new == gamma

            my_base_lr = cfg._cfg_dict['my_base_lr']
            wd = cfg._cfg_dict['wd']
            optimizer = cfg._cfg_dict['optimizer']

            run_time = datetime_log_fancy()
            work_dir = f'models/N{n_ways}-K{k_shots}-B4 DCL-FT LR {my_base_lr} WD {wd} {optimizer["type"]} StepG {gamma_new}'
            if os.path.exists(work_dir):
                print('DIR exits', work_dir)
                continue
            assert os.path.exists('models')
            assert not os.path.exists(work_dir)
            cfg._cfg_dict['work_dir'] = work_dir
            print(work_dir)

            print(cfg.lr_config)
            main(cfg)

            from time import sleep

            print('Take a rest', datetime_now())
            sleep(60)
            print('Finish', datetime_now())
