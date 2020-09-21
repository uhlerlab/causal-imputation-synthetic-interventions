from filenames import load_cell_ranks, load_pert_ranks
import pandas as pd
import itertools as itr
import ipdb


def get_data_block(cell_start: int, cell_stop: int, pert_start: int, pert_stop: int, name='level2_filtered'):
    print("[get_data_block] loading averages")
    averages = pd.read_pickle(f'data/processed/averages/{name}.pkl')
    pert_ranks = load_pert_ranks()
    cell_ranks = load_cell_ranks()
    cells = set(cell_ranks[cell_start:cell_stop].index)
    perts = set(pert_ranks[pert_start:pert_stop].index) | {'DMSO'}

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