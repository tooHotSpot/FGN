from cp_utils.cp_time import datetime_log_fancy

my_base_lr = 0.005
wd = 0.00005
optimizer = dict(
    # type='SGD', momentum=0.9, nesterov=True,
    # type='Adam',
    type='Adagrad',
    lr=my_base_lr,
    weight_decay=wd,
    paramwise_cfg=dict(custom_keys={
        # 'backbone': dict(lr_mult=1.0, decay_mult=1.0),
        # 'rpn_head': dict(lr_mult=1.0, decay_mult=1.0),
        'roi_head': dict(lr_mult=0.1, decay_mult=1.0)
        # 'bbox_head': dict(lr_mult=0.01, decay_mult=1.0),
    })
)

# optimizer_config = dict(
#     type='GradientCumulativeOptimizerHook',
#     cumulative_iters=4,
#     grad_clip=None
# )

optimizer_config = dict(type='OptimizerHook', grad_clip=None)

# Learning policy
lr_config = dict(
    # Fixed, Step, CosineAnnealing, etc.
    # policy='Fixed',
    # policy='CosineAnnealing', min_lr_ratio=0.01,
    policy='Step', step=[3], gamma=0.1, min_lr=0.000001, by_epoch=True,
    # policy='Fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01
)

# Set another dir if required
# work_dir = None
# checkpoint = None

# run_time = datetime_log_fancy()
# run_name = '...'
# work_dir = f'models/{run_time}_{run_name}'
work_dir = None

# Add a checkpoint to continue model training or validate a train model
# upper = '/home/neo/PycharmProjects/Course1/subprojects/sp02_omniiseg_fgn_mmdet/'
# checkpoint = upper + work_dir + '/epoch_6.pth'
checkpoint = None

# Runtime settings
checkpoint_config = dict(
    interval=1,
    save_optimizer=True,
    # Keep all big models
    max_keep_ckpts=6,
    by_epoch=True
)

log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

runner = dict(
    type='EpochBasedRunner',
    max_epochs=6
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# Possible: dict(type='EmptyCacheHook'), dict(type='IterTimerHook').
custom_hooks = []
