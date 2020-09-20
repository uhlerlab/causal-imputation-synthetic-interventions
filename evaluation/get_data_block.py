from filenames import load_cell_ranks, load_pert_ranks
import pandas as pd
import ipdb


def get_data_block(cell_start: int, cell_stop: int, pert_start: int, pert_stop: int, name='level2_filtered'):
    print("[get_data_block] loading averages")
    averages = pd.read_pickle(f'data/processed/averages/{name}.pkl')
    pert_ranks = load_pert_ranks()
    cell_ranks = load_cell_ranks()
    cells = set(cell_ranks[cell_start:cell_stop].index)
    perts = set(pert_ranks[pert_start:pert_stop].index)
    block = averages.query("cell_id in @cells and pert_id in @perts")
    return block, cells, perts


if __name__ == '__main__':
    data_block = get_data_block(0, 10, 0, 20)


