import os
import numpy as np
from cp_utils.cp_dir_file_ops import check_file_if_exists, read_json, write_json_safe

from datasets.fewshotiseg.base_fst import BaseFewShotISEG


def select_indices(ds: BaseFewShotISEG):
    prefix = f'{ds.setup}_' \
             f'{ds.sampling_origin_ds}_' \
             f'{ds.sampling_origin_ds_subset}_' \
             f'{ds.sampling_cats}_' \
             f'K{ds.k_shots}_'

    FGN_K = 3 * ds.k_shots
    min_required = FGN_K
    max_required = FGN_K + 1
    if ds.sampling_cats == 'novel':
        print('For novel categories choose only', FGN_K, 'instances')
        FGN_K = ds.k_shots
        # Adding one is required to omit bad training scenario
        min_required = FGN_K + 1
        max_required = FGN_K + 2

    file_fp = os.path.join(ds.root, prefix + f'FINETUNE_REAL_INDICES.json')
    print('Checking FINETUNE INDEXES in', file_fp)
    if not check_file_if_exists(file_fp):
        if len(ds.cat_ids) == 0:
            print('Dataset instance does not have required arrays')
            raise NotImplementedError

        # Create an array for all images
        presence_arr = np.zeros((len(ds.cat_ids), ds.cats_total_amount), dtype=np.int32)
        for i in range(len(ds.cat_ids)):
            np.add.at(presence_arr[i], ds.cat_ids[i], 1)

        # 1. Select a subset of images which does not have cats_to_del_ at all
        cats_to_del__in = np.array(presence_arr[:, ds.cats_to_del_]).sum(axis=-1)
        imgs_no_cats_to_del_ = np.where(cats_to_del__in == 0)[0]
        print('Amount of images with no categories to delete', len(imgs_no_cats_to_del_))

        # 2. Sort selected images by the amount of instances on an image (in descending order)
        total_objects_amount = presence_arr[imgs_no_cats_to_del_, :].sum(axis=-1)
        indices = np.argsort(total_objects_amount)[::-1]
        imgs_no_cats_to_del__desc = imgs_no_cats_to_del_[indices]
        print('Amount of instances of categories to save for each image')
        print(presence_arr[imgs_no_cats_to_del__desc].sum(axis=-1))

        # 3. Remove images with no objects annotated and with more than FGN_K objects of the same class
        indices = np.nonzero(
            (presence_arr[imgs_no_cats_to_del__desc].sum(axis=-1)) *
            (presence_arr[imgs_no_cats_to_del__desc].max(axis=-1) <= FGN_K)
        )
        imgs_no_cats_to_del__desc_non_zero = imgs_no_cats_to_del__desc[indices]
        imgs_pool = presence_arr[imgs_no_cats_to_del__desc_non_zero]
        print('The same as the last but after filtration of by [0; 3K] borders')
        print(imgs_pool.sum(axis=-1))

        # 4. Check that images in a selection do not have cats_to_del_ examples (and crowd examples also)
        selection = imgs_pool[:, ds.cats_to_del_]
        print('Images pool array shape with cats_to_del_', selection.shape)
        print('Min / max cats_to_del_ instances amount', selection.min(), selection.max())
        del selection

        # 5. Check that images in a selection have all cats_to_save with at least K examples for each category
        cats_to_save_only_presence_vec = imgs_pool[:, ds.cats_to_save].sum(axis=0)
        print('Amount of instances of cats_to_save in a subset')
        print(cats_to_save_only_presence_vec)
        cat_to_save_no_3K = np.where(cats_to_save_only_presence_vec < FGN_K)[0]
        print(f'Critical cats with less than {FGN_K} instances:', cat_to_save_no_3K)

        # 6. For each category, check amount of images with 1 instance, 2 instances, ...
        for cat_id in ds.cats_to_save:
            column = imgs_pool[:, cat_id]
            appear = {}
            for i in range(1, max(column) + 1):
                appear[i] = np.count_nonzero(column == i)
            # appear = dict(zip(*np.unique(column, return_counts=True)))
            print(cat_id, appear)
            del appear
        del cat_id
        # Seems that there are a lot images with 1 and 2 instances

        # 7. Estimating how difficult it may be to fold the selection
        # 7.1. Count how many instances do images across the dataset contain
        unique_cats_ids_on_img = (imgs_pool > 0).sum(axis=-1)
        unique_cats_ids_amount, unique_cats_ids_amount_counts = np.unique(unique_cats_ids_on_img, return_counts=True)
        print('Instances, images with this amount of instances')
        print(np.array(list(zip(unique_cats_ids_amount, unique_cats_ids_amount_counts))))

        # 7.2. For each category, count images where only 1 cat
        indices_only_1cat = np.where((imgs_pool > 0).sum(axis=-1) == 1)[0]
        print('Amount of images with only 1 category:', len(indices_only_1cat))

        selection = imgs_pool[indices_only_1cat]
        print('Selection shape', selection.shape)

        for cat_id in ds.cats_to_save:
            column = selection[:, cat_id]
            unique, counts = np.unique(column, return_counts=True)
            print(f'Cat {cat_id:2}', dict(zip(unique, counts)))
            del unique, counts
        del selection

        # 8. Sort by the amount of representation
        imgs_set_hidden_indexes = set()
        total_cats_insts_in_set = np.zeros(ds.cats_total_amount)
        total_cats_insts_in_set[ds.cats_to_del_] = -1
        # For same selection on every generation
        order = np.argsort(cats_to_save_only_presence_vec, kind='stable')
        cats_to_save_ascending_cat_ids = ds.cats_to_save[order]

        for n, cat_id in enumerate(cats_to_save_ascending_cat_ids):
            if all(total_cats_insts_in_set[ds.cats_to_save] >= max_required):
                print('Already finished for all cats')
                break
            success = False
            # Select images which have this category
            selection = imgs_pool[:, cat_id]
            hidden_indices = np.where(selection != 0)[0]
            real_indices = imgs_no_cats_to_del__desc_non_zero[hidden_indices]
            print(f'Cat {cat_id} total images with this cat', len(imgs_pool[hidden_indices]))

            selected_num_this_cat_examples = imgs_pool[hidden_indices][:, cat_id]
            selected_num_each_cat_examples = (imgs_pool[hidden_indices] > 0).sum(axis=-1)
            selected_triple = np.array(list(zip(selected_num_this_cat_examples,
                                                selected_num_each_cat_examples,
                                                hidden_indices)))
            groups = {amount: [] for amount in np.unique(selected_num_this_cat_examples)}
            for amount in groups:
                amount_group_indices = np.where(selected_triple[:, 0] == amount)[0]
                amount_group = selected_triple[amount_group_indices, :]
                if len(amount_group) > 1:
                    order = np.argsort(amount_group[:, 1])
                    amount_group = amount_group[order]
                groups[amount] = amount_group

            # Check that all images with these indices have this category
            check_required = False
            for real_index in real_indices:
                cat_ids = ds.cat_ids[real_index]
                assert cat_id in cat_ids
                if not check_required:
                    print('Checked ONE image and everything is OK')
                    break
            if check_required:
                print('Checked ALL images and everything is OK')

            # Perform a selection
            triples_selected = []
            for amount in sorted(groups):
                print('Trying amount', amount)
                if total_cats_insts_in_set[cat_id] >= max_required:
                    print('Already finished for cat', cat_id)
                    break
                if total_cats_insts_in_set[cat_id] + amount > max_required:
                    print('Amount', total_cats_insts_in_set[cat_id] + amount,
                          'exceeds the max_required limit', amount)
                    continue
                    # while total_cats_insts_in_set[cat_id] + amount > total_required:
                    #     if len(triples_selected) == 0:
                    #         raise NotImplementedError
                    #     # Two strategies: random and try to delete less
                    #     # 1
                    #     triple_index = np.random.choice(len(triples_selected), replace=False)
                    #     # 2
                    #     triple_index = 0
                    #     triple = triples_selected[triple_index]
                    #     hidden_img_index = triple[2]
                    #     imgs_set_hidden_indexes.remove(hidden_img_index)
                    #     total_cats_insts_in_set -= imgs_pool[hidden_img_index]
                    #     del triples_selected[triple_index]
                    #     print('Deleted triple', triple)

                for triple in groups[amount]:
                    _, _, hidden_img_index = triple
                    cur_img_cats = imgs_pool[hidden_img_index]
                    assert cur_img_cats[cat_id] == amount
                    summary = total_cats_insts_in_set + cur_img_cats
                    if max(summary) > max_required:
                        print('More instances than required', summary[ds.cats_to_save])
                        continue
                    else:
                        total_cats_insts_in_set = summary
                        imgs_set_hidden_indexes.add(hidden_img_index)
                        triples_selected.append(triple)
                        print('Added ', triple, 'new summary', summary[ds.cats_to_save])
                        if min_required <= total_cats_insts_in_set[cat_id] <= max_required:
                            success = True
                            print('Chosen successfully', total_cats_insts_in_set[ds.cats_to_save])
                            break
                if not success:
                    print('Could not find an appropriate subset')

        print('*** Finished for all cats *** ')

        print('Total images in the set', len(imgs_set_hidden_indexes))
        print('Amount of class instances depicted\n', total_cats_insts_in_set[ds.cats_to_save])
        imgs_list_hidden_indexes = list(imgs_set_hidden_indexes)
        real_indices = imgs_no_cats_to_del__desc_non_zero[imgs_list_hidden_indexes]
        # print('Instances amount for each category, sums have to be equal')
        a: np.ndarray = imgs_pool[imgs_list_hidden_indexes].sum(axis=0)
        b: np.ndarray = presence_arr[real_indices].sum(axis=0)
        assert all(a == b)

        write_json_safe(file_fp, real_indices.astype(np.int32).tolist())
        print('Saved real_indices array to file:', file_fp)
    else:
        real_indices = read_json(file_fp)
        real_indices = np.array(real_indices, dtype=np.int32)
        print('Read real_indices array from file:', file_fp)
    return real_indices
