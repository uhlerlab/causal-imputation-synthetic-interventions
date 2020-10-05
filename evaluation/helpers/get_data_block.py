from filenames import load_cell_ranks, load_pert_ranks, load_cell_ranks_common, load_pert_ranks_common
from typing import Optional
import pandas as pd
import itertools as itr
import ipdb
import random


def remove_celltypes_no_control(
        data,
        control_pert='DMSO'
):
    retained_units = set(data.query("intervention == @control_pert").index.get_level_values('unit'))
    retained_data = data.query("unit in @retained_units")
    ipdb.set_trace()
    return retained_data


def get_data_block(
        num_cells: Optional[int] = 10,
        num_perts: Optional[int] = 20,
        cell_start: Optional[int] = None,
        pert_start: Optional[int] = None,
        name='level2_filtered',
        common=True
):
    print("[get_data_block] loading averages")
    averages = pd.read_pickle(f'data/processed/averages/{name}.pkl')
    if common:
        pert_ranks = load_pert_ranks_common()
        cell_ranks = load_cell_ranks_common()
    else:
        pert_ranks = load_pert_ranks()
        cell_ranks = load_cell_ranks()

    if cell_start is not None:
        cell_stop = cell_start+num_cells if num_cells is not None else len(cell_ranks)
        cells = set(cell_ranks[cell_start:cell_stop].index)
    else:
        random.seed(123)
        cells = set(random.sample(list(cell_ranks.index), num_cells))

    if pert_start is not None:
        pert_stop = pert_start+num_perts if num_perts is not None else len(pert_ranks)
        perts = set(pert_ranks[pert_start:pert_stop].index) | {'DMSO'}
    else:
        random.seed(123)
        perts = set(random.sample(list(pert_ranks.index), num_perts)) | {'DMSO'}

    block = averages.query("cell_id in @cells and pert_id in @perts")
    block.index.rename(['unit', 'intervention'], inplace=True)
    missing = pd.MultiIndex.from_tuples(set(itr.product(cells, perts)) - set(block.index), names=['unit', 'intervention'])
    return block, cells, perts, missing


if __name__ == '__main__':
    data_block, cells, perts, targets = get_data_block(10, 20, 1000, 1020)
    from src.algorithms2 import impute_unit_mean, impute_intervention_mean, impute_two_way_mean

    m = impute_unit_mean(data_block, targets)
    m2 = impute_intervention_mean(data_block, targets)
    m3 = impute_two_way_mean(data_block, targets)

    # test: verify same unit gets same values
    unit_means = m.groupby('unit').mean()
    unit_passes = unit_means.eq(m).all().all()
    print(f"Unit passes: {unit_passes}")

    # test: verify same intervention gets same values
    iv_means = m2.groupby('intervention').mean()
    iv_passes = iv_means.eq(m2).all().all()
    print(f"iv passes: {iv_passes}")
