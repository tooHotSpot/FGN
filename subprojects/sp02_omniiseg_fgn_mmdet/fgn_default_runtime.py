from cp_utils.cp_time import datetime_log_fancy

# Set another dir if required
# work_dir = 'models/{experiment_name}_{run_name}_{comment}'
work_dir = 'models/' + datetime_log_fancy() + '_Train_COCO'

# Runtime settings
# Saving redundant models is a true waste of SSD health
checkpoint_config = dict(
    interval=1,
    out_dir='models',
    save_optimizer=True,
    max_keep_ckpts=1,
    by_epoch=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

custom_hooks = [
    dict(type='EmptyCacheHook'),
    dict(type='IterTimerHook'),
]

log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

