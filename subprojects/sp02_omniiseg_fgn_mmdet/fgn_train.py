_base_ = [
    'fgn_r50_c4_densecl.py',
    # 'fgn_r50_c4_scratch.py',
    'fgn_train_schedule.py',
]

_train = 'train'
_val = 'val'
_parents = 'parents'
_children = 'children'
_ignore = 'Ignore'
_base = 'base_'
_novel = 'novel'
_all = 'all'

train_ds_cfg = dict(
    n_ways=3,
    k_shots=3,
    verbose=False,
    ds_base_='COCO',
    ds_base__subset=_train,
    ds_novel='VOC',
    ds_novel_subset=_val,
    sampling_origin_ds='COCO',
    sampling_origin_ds_subset=_train,
    sampling_cats=_base,
    first_parents__only=0,
    first_children_only=0,
    augment_qry=True,
    augment_spp=True,
    # Too many children and some images may appear more times than other
    sampling_scenario=_parents,
    repeats=1,
    shuffle=True,
    qry_cats_choice_random=True,
    qry_cats_order_shuffle=True,
    spp_random=True,
    delete_qry_insts_in_spp_insts_on_train=True,
    finetune=_ignore,
    spp_fill_ratio=0.8,
    batch=8
)

# Set common validation options
eval_ds_cfg_gen = train_ds_cfg.copy()
eval_ds_cfg_gen['sampling_origin_ds_subset'] = _val
eval_ds_cfg_gen['sampling_scenario'] = _parents
eval_ds_cfg_gen['first_children_only'] = 0
eval_ds_cfg_gen['qry_cats_choice_random'] = True
eval_ds_cfg_gen['qry_cats_order_shuffle'] = True
eval_ds_cfg_gen['spp_random'] = True
eval_ds_cfg_gen['shuffle'] = False
eval_ds_cfg_gen['repeats'] = 1
eval_ds_cfg_gen['augment_qry'] = False
eval_ds_cfg_gen['augment_spp'] = False
eval_ds_cfg_gen['batch'] = 4

# Define validation datasets
eval_ds_cfg0 = eval_ds_cfg_gen.copy()
eval_ds_cfg0['sampling_cats'] = _base

# eval_ds_cfg1 = eval_ds_cfg_gen.copy()
# eval_ds_cfg1['sampling_cats'] = _novel

# eval_ds_cfg2 = eval_ds_cfg_gen.copy()
# eval_ds_cfg2['sampling_cats'] = _all

del eval_ds_cfg_gen
