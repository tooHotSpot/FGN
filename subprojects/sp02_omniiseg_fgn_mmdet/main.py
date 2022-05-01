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

# if debugger_is_active():
#     import os
#     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

workers = 0 if define_env() == 'PC' or debugger_is_active() else 4
pin_memory_start = False if define_env() == 'PC' or debugger_is_active() else True
persistent_workers = False if define_env() == 'PC' or debugger_is_active() else True

eval_hooks_list: List[BaseEvalHook] = []

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from torch.utils.data._utils.collate import default_collate


def collate_fn_new(batch):
    # Delete varying sequences from the batch
    keys = ['qry_cat_ids_real', 'qry_cat_ids', 'qry_bboxes', 'qry_isegmaps']
    batch_add = {key: [torch.as_tensor(batch[i][key])
                       for i in range(len(batch))]
                 for key in keys}

    for i in range(len(batch)):
        for key in keys:
            batch[i].pop(key, None)

    batch = default_collate(batch)
    # Add backward to batch as lists of tensors
    batch.update(batch_add)
    return batch


class OptEpochBasedRunner(EpochBasedRunner):
    model: FGN
    data_loader: Optional[DataLoader]
    tb_hook: TensorboardLoggerHook
    tb_flags_unlogged = True

    def __init__(self, model, work_dir, **kwargs):
        print('Model work_dir', work_dir)
        if check_dir_if_exists(work_dir):
            print('-> Already exists')
        else:
            create_empty_dir_safe(work_dir)
            print('-> Created new')
        super(OptEpochBasedRunner, self).__init__(model=model, work_dir=work_dir, **kwargs)

    def run_iter(self, data_batch, train_mode, **kwargs):
        t1 = datetime_now()
        super(OptEpochBasedRunner, self).run_iter(data_batch, train_mode, **kwargs)
        t_total = datetime_diff_ms(t1)
        self.tb_hook.writer.add_scalar('Time/TrainStep', t_total, self.iter)
        if self.tb_flags_unlogged:
            self.tb_flags_unlogged = False
            _cfg = self.model.cfg
            hparam_dict = {
                'Optimizer': _cfg.optimizer.type,
                'Policy': _cfg.lr_config.type,
                'Warmup': _cfg.lr_config.warmup,
                'WarmupIters': _cfg.lr_config.warmup_iters,
                # 'CumulativeSteps': _cfg.optimizer_config.cumulative_iters,
                'Workers': workers,
                'PinMemory': pin_memory_start,
                'FP16': self.model.fp16_enabled,
                'PersistentWorkers': persistent_workers,
            }
            del _cfg
            texts = [str(key) + ': ' + str(hparam_dict[key]) + '\n'
                     for key in sorted(hparam_dict)]
            text = ''.join(texts)
            self.tb_hook.writer.add_text('Settings', text, 0)
            print('OptEpochBasedRunner: tried to add text')

        return

    def log_hyperparams(self):
        # Log learning rate for Backbone with RPN and Heads
        lrs = []
        momentums = []
        wds = []
        for group in self.optimizer.param_groups:
            if group['lr'] not in lrs:
                lrs.append(group['lr'])
            # if group['momentum'] not in momentums:
            #     momentums.append(group['momentum'])
            if group['weight_decay'] not in wds:
                wds.append(group['weight_decay'])

        lrs = sorted(lrs, reverse=True)
        print('OptEpochBasedRunner: Total different lrs', len(lrs))
        print('Values LR', lrs)
        if len(lrs) == 1:
            lrs = [lrs[0], lrs[0]]
        print('Values WD', wds)
        if len(wds) == 1:
            wds = [wds[0], wds[0]]

        for i in range(len(lrs)):
            self.tb_hook.writer.add_scalar(f'Hyperparams/LR{i}', lrs[i], self.iter)
        for i in range(len(wds)):
            self.tb_hook.writer.add_scalar(f'Hyperparams/WD{i}', wds[i], self.iter)

        # for i in range(len(momentums)):
        #     self.tb_hook.writer.add_scalar(f'Hyperparams/Momentum{i}', momentums[i], self.iter)
        self.tb_hook.writer.flush()

    def check(self):
        self.model.eval()
        print('*' * 50, ' OptEpochBasedRunner: Tried to evaluate again! ', '*' * 50)
        print('WorkDir', self.work_dir)
        try:
            print('OptEpochBasedRunner: Saving a checkpoint at an iter', self.iter)
            sp = f'ModelLatest.{datetime_log_fancy()}.Epoch_{self.epoch}.Iter_{self.iter}.pth'
            self.save_checkpoint(out_dir=self.work_dir,
                                 filename_tmpl=sp,
                                 save_optimizer=True,
                                 create_symlink=False)
            print('OptEpochBasedRunner: Saved model to the file', sp)
            print('OptEpochBasedRunner: Finished validation OK', datetime_log_fancy())
        except Exception as error:
            warnings.warn('Did not save this time, something is wrong', UserWarning)
            logger.exception(error)
        print('Evaluation over eval_hooks_list with of size', len(eval_hooks_list))
        try:
            for eval_hook_cur in eval_hooks_list:
                # noinspection PyProtectedMember
                eval_hook_cur._do_evaluate(runner=self)
            print('OptEpochBasedRunner: Finished validation OK', datetime_log_fancy())
        except Exception as error:
            warnings.warn('Did not evaluate this time, something is wrong', UserWarning)
            logger.exception(error)

    def log_iter_info(self):
        if not define_env() == 'SERVER':
            return

        # RPN Amount of valid anchors for each image
        # RPN Amount of positive / negative ROIs
        # RPN Separate losses (To compare to the general loss)
        wr = self.tb_hook.writer
        rh = self.model.rpn_head

        wr.add_scalar(f'rpn0_loss/loss_cls', rh.rpn_log_loss_cls, self.iter)
        wr.add_scalar(f'rpn0_loss/loss_bbox', rh.rpn_log_loss_bbox, self.iter)
        wr.add_scalar(f'rpn3_num_total_pos_neg/num_total_pos', rh.rpn_log_num_total_pos, self.iter)
        wr.add_scalar(f'rpn3_num_total_pos_neg/num_total_neg', rh.rpn_log_num_total_neg, self.iter)
        for i in range(self.model.n_ways):
            wr.add_scalar(f'rpn1_inds/n_pos_inds_n{i:02}', rh.rpn_log_n_pos_inds[i], self.iter)
            wr.add_scalar(f'rpn1_inds/n_neg_inds_n{i:02}', rh.rpn_log_n_neg_inds[i], self.iter)
            wr.add_scalar(f'rpn2_n_anchors/n_anchors_n{i:02}', rh.rpn_log_n_anchors[i], self.iter)
            wr.add_scalar(f'rpn2_n_anchors/n_achors_valid_n{i:02}', rh.rpn_log_n_anchors_valid[i], self.iter)
            # Show feature maps
        del rh

    def train(self, ds: BaseFewShotISEG, **kwargs):
        print('OptEpochBasedRunner: Starting a train epoch')
        if self.epoch > 0:
            ds.reshuffle()
        dataloader = DataLoader(ds, batch_size=ds.batch,
                                num_workers=workers, pin_memory=pin_memory_start,
                                prefetch_factor=2, persistent_workers=persistent_workers,
                                collate_fn=collate_fn_new)
        self.model.ds = ds
        self.data_loader = dataloader
        t1 = datetime_now()
        self.model.train()
        self.mode = 'train'
        self._max_iters = self._max_epochs * len(dataloader)
        self.call_hook('before_train_epoch')
        print('Changing the learning rate')
        self.log_hyperparams()
        # Prevent possible deadlock during epoch transition
        time.sleep(5)
        part = max(2000, len(dataloader) // 8)

        t1 = datetime_now()
        for i, data_batch in enumerate(dataloader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self.log_iter_info()
            self._iter += 1
            if i % part == 0 and i > 0:
                if len(dataloader) - i < part:
                    print(f'Batch {i} not evaluating, part {part}')
                    continue
                print('Total part time', datetime_diff(t1))
                t1 = datetime_now()
                self.model.eval()
                self.check()
                self.model.train()

        print('OptEpochBasedRunner: Calling hook after_train_epoch')
        self.call_hook('after_train_epoch')
        self._epoch += 1
        print('OptEpochBasedRunner: Finishing epoch', self.epoch)
        self.model.ds = None
        self.data_loader = None
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        print('OptEpochBasedRunner: Deleted dataloader and cleaned CUDA')
        t_total_val = datetime_diff_ms(t1)
        t_total_str = datetime_diff(t1)
        print('OptEpochBasedRunner: Total epoch time', t_total_str)
        self.tb_hook.writer.add_scalar('Time/TrainEpoch', t_total_val, self.iter)
        print(f'OptEpochBasedRunner: Finished train '
              f'Epoch {self.epoch} / {self.max_epochs}', datetime_log_fancy())
        print('WorkDir', self.work_dir)


class OptEvalHook(BaseEvalHook):
    dataset: BaseFewShotISEG

    def __init__(self, dataset: BaseFewShotISEG, **kwargs):
        self.dataset = dataset
        dataloader = DataLoader(dataset)
        super(OptEvalHook, self).__init__(dataloader, **kwargs)
        del self.dataloader
        self.dataloader = None

    def _do_evaluate(self, runner: OptEpochBasedRunner):
        """ Perform evaluation and save checkpoint. """
        if not self._should_evaluate(runner):
            return

        ds: BaseFewShotISEG = self.dataset
        # self.dataset.get_plot = True

        print('OptEvalHook: Performing an evaluation of', ds.suffix)
        dataloader = DataLoader(
            ds,
            batch_size=ds.batch,
            num_workers=workers,
            pin_memory=pin_memory_start,
            prefetch_factor=2,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn_new
        )

        t1 = datetime_now()
        results = []
        results_pkl_dir_sp = 'ResultsChunked'
        results_pkl_dir_fp = os.path.join(runner.model.cfg.work_dir, results_pkl_dir_sp)
        create_empty_dir_unsafe(results_pkl_dir_fp)

        print('OptEvalHook: Launching evaluation')
        tenth = max(len(dataloader) // 10, 1)
        pbar = tqdm(total=len(dataloader))
        counter = 0
        for i, data in enumerate(dataloader):
            # Using torch.no_grad() for simple_test(...)
            result = runner.model.simple_test(**data, rescale=True)
            if len(result) == 0:
                result = [result]
            results.extend(result)

            if len(results) == 1000 or i == len(dataloader) - 1:
                result_file_fp = os.path.join(results_pkl_dir_fp, f'{counter:02}.pkl')
                counter += 1
                write_pkl_safe(result_file_fp, data=results)
                results = []
            if i % tenth == 0 and i > 0:
                pbar.update(tenth)
                # print(f'> Memory usage {give_mem()}%')

        pbar.close()

        t2 = datetime_now()
        print('OptEvalHook: Finishing')
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        print('OptEvalHook: Deleted dataloader and cleaned CUDA')

        general_tag = f'{ds.sampling_origin_ds}_' \
                      f'{ds.sampling_origin_ds_subset}_' \
                      f'FT_{ds.finetune}'
        metrics = ds.evaluate(results=None, results_pkl_dir_fp=results_pkl_dir_fp,
                              model_dir=runner.model.cfg.work_dir)

        for key in metrics:
            tag = f'{general_tag}/{key}_{ds.sampling_cats}'
            if ds.sampling_scenario == 'children':
                tag = f'{general_tag}/{key}_{ds.sampling_cats}_{ds.sampling_scenario}'
            runner.tb_hook.writer.add_scalar(tag, metrics[key], runner.iter)

        # self.key_indicator = 'isegm_mAP'
        # key_score = eval_res[self.key_indicator]
        # self._save_ckpt(runner, key_score)

        t_total = datetime_diff(t1, t2)
        print('Total evaluation time', t_total)
        t_total = datetime_diff_ms(t1, t2)
        runner.tb_hook.writer.add_scalar('Time/Evaluation', t_total, runner.iter)
        runner.tb_hook.writer.flush()
        print('WorkDir', runner.work_dir)


def init_ds_class_by_config(config):
    name = config['sampling_origin_ds']

    if name not in ('OMNIISEG', 'MNISTISEG', 'VOC', 'COCO'):
        print('Dataset name not in list', name)
        assert False

    if name == 'MNISTISEG':
        return MNISTFewShotISEG(config)
    elif name == 'OMNIISEG':
        return OMNIFewShotISEG(config)
    elif name == 'COCO':
        return COCOFewShot(config)
    elif name == 'VOC':
        return VOCFewShot(config)


def main(cfg):
    device = 'cpu' if define_env() == 'PC' else 'cuda:0'
    print('Using device', device)
    # base_only = 0
    # novel_only = 0
    # base_novel = 0
    # In total 80 categories

    if 'ft_ds_cfg0' in cfg:
        ft_ds0 = init_ds_class_by_config(config=cfg.get('ft_ds_cfg0'))
        print('Finetune DS 0 #Length', len(ft_ds0))
        # ft_ds0.visualize(n_imgs=20, vis_dir_sp='ft_ds0_vis_dir')

        ft_ds1 = init_ds_class_by_config(config=cfg.get('ft_ds_cfg1'))
        print('Finetune DS 1 #Length', len(ft_ds1))
        # ft_ds1.visualize(n_imgs=20, vis_dir_sp='ft_ds1_vis_dir')

        ft_ds0.merge_ds(ft_ds1)
        print('Merged FT DS  #Length', len(ft_ds0))
        # ft_ds0.visualize(n_imgs=20, vis_dir_sp='merged_ds_vis_dir')
        train_ds = ft_ds0
    else:
        train_ds = init_ds_class_by_config(config=cfg.get('train_ds_cfg'))
        # train_ds.visualize(n_imgs=10, choose_random=True, vis_dir_sp='train_ds_before_merge')
        print('Train DS 0 #Length', len(train_ds))

    model: FGN = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )

    assert FGN.n_ways == train_ds.n_ways, f'FGN N {FGN.n_ways} != DS N {train_ds.n_ways}'
    assert FGN.k_shots == train_ds.k_shots, f'FGN K {FGN.k_shots} != DS K {train_ds.k_shots}'
    if train_ds is not None:
        assert FGN.n_ways == train_ds.n_ways
        assert FGN.k_shots == train_ds.k_shots

    # Remove the last res4 layer when initialized from a pretrained backbone
    if model.backbone.frozen_stages != -1:
        model.backbone.res_layers = model.backbone.res_layers[:-1]
        model.backbone.eval()

    # Save the config in the model for convenience
    model.cfg = cfg
    checkpoint = cfg.checkpoint
    meta = None
    model.to(device)
    optimizer: torch.optim.Optimizer = build_optimizer(model, cfg.optimizer)

    logger = get_root_logger(log_level=cfg.log_level)

    runner: OptEpochBasedRunner
    runner = OptEpochBasedRunner(
        model,
        batch_processor=None,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=None,
        max_epochs=cfg.runner.max_epochs,
    )
    if checkpoint is not None:
        runner.model.to(device)
        runner.resume(checkpoint, resume_optimizer=True, map_location=device)
        print('Loaded Runner Meta', runner.meta, 'Epoch', runner.epoch, 'Iter', runner.iter)
        print('Initialization from checkpoint OK')
    else:
        # with contextlib.redirect_stdout(None):
        model.init_weights()
        print('Initialization from scratch OK')

    model.to(device)
    dsets = [train_ds]

    print(cfg.lr_config)

    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config,
        custom_hooks_config=cfg.custom_hooks
    )

    runner.tb_hook = TensorboardLoggerHook(log_dir=cfg.work_dir)
    runner.tb_hook.before_run(runner)
    runner.register_hook(runner.tb_hook, priority='NORMAL')

    keys = []
    for key in cfg._cfg_dict.keys():
        if 'eval_ds_cfg' in key and key[-1].isnumeric():
            keys.append(key)

    global eval_hooks_list
    eval_hooks_list = []
    for key in sorted(keys):
        print('Creating a dataset from config', key)
        eval_ds = init_ds_class_by_config(config=cfg.get(key))
        assert eval_ds.n_ways == model.n_ways, f'FGN N {model.n_ways} != DS N {eval_ds.n_ways}'
        assert eval_ds.k_shots == model.k_shots, f'FGN N {model.n_ways} != DS N {eval_ds.n_ways}'
        eval_hook = OptEvalHook(
            # interval=5,
            dataset=eval_ds,
            # For models trained in 10 epochs
            start=0,
            by_epoch=True,
            save_best=None,
            greater_keys=['segm_mAP'],
        )
        eval_hooks_list.append(eval_hook)
        runner.register_custom_hooks(eval_hook)

    print('Totally added ', len(eval_hooks_list), 'eval hooks')
    runner.get_hook_info = lambda: 'Hook info was hidden manually'

    to_train = True
    if to_train:
        runner.run(dsets, cfg.workflow)
    else:
        runner.check()


if __name__ == '__main__':
    config_file = 'fgn_train.py'
    cfg = Config.fromfile(config_file)
    n_ways = cfg._cfg_dict['train_ds_cfg']['n_ways']
    k_shots = cfg._cfg_dict['train_ds_cfg']['k_shots']
    batch = 8
    if n_ways == 1 and k_shots == 1:
        batch = 12
    elif n_ways == 3 and k_shots == 1:
        batch = 10
    elif n_ways == 3 and k_shots == 3:
        batch = 8
    cfg._cfg_dict['train_ds_cfg']['batch'] = batch
    print('BATCH SIZE', batch)
    main(cfg)
