_base_ = [
    'fgn_r50_c4_densecl.py',
    # 'fgn_r50_c4_scratch.py',
    'fgn_ft_schedule.py',
]

_train = 'train'
_val = 'val'
# For MNIST
_test = 'test'
# For COCO
_trainval = 'trainval'
_parents = 'parents'
_children = 'children'
_ignore = 'Ignore'
_select = 'Select'
_use = 'Use'
_base = 'base_'
_novel = 'novel'
_all = 'all'

ft_ds_cfg0 = dict(
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
    sampling_scenario=_children,
    # Since the FT dataset is small
    repeats=10,
    shuffle=True,
    qry_cats_choice_random=True,
    qry_cats_order_shuffle=True,
    spp_random=True,
    delete_qry_insts_in_spp_insts_on_train=True,
    # ///////////////////////////////////////////////////// Select
    finetune=_select,
    spp_fill_ratio=0.8,
    batch=4
)

ft_ds_cfg1 = dict(
    n_ways=3,
    k_shots=3,
    verbose=False,
    ds_base_='COCO',
    ds_base__subset=_train,
    ds_novel='VOC',
    ds_novel_subset=_val,
    sampling_origin_ds='VOC',
    sampling_origin_ds_subset=_trainval,
    sampling_cats=_novel,
    first_parents__only=0,
    first_children_only=0,
    augment_qry=True,
    augment_spp=True,
    # Too many children and some images may appear more times than other
    sampling_scenario=_children,
    repeats=1,
    shuffle=True,
    qry_cats_choice_random=True,
    qry_cats_order_shuffle=True,
    spp_random=True,
    delete_qry_insts_in_spp_insts_on_train=True,
    # ///////////////////////////////////////////////////// Select
    finetune=_select,
    spp_fill_ratio=0.8,
    batch=1
)

eval_ds_cfg0 = dict(
    n_ways=3,
    k_shots=3,
    verbose=False,
    ds_base_='COCO',
    ds_base__subset=_train,
    ds_novel='VOC',
    ds_novel_subset=_val,
    sampling_origin_ds='VOC',
    sampling_origin_ds_subset=_trainval,
    sampling_cats=_novel,
    first_parents__only=0,
    first_children_only=0,
    # Do not augment
    augment_qry=False,
    augment_spp=False,
    # Too many children and some images may appear more times than other
    sampling_scenario=_children,
    repeats=1,
    shuffle=False,
    qry_cats_choice_random=True,
    qry_cats_order_shuffle=True,
    spp_random=True,
    delete_qry_insts_in_spp_insts_on_train=True,
    # ///////////////////////////////////////////////////// Use
    finetune=_use,
    spp_fill_ratio=0.8,
    batch=4
)
