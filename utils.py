import pandas as pd
from sklearn.preprocessing import minmax_scale
import ipdb
import itertools as itr
from collections import defaultdict


def optional_str(s, predicate):
    return s if predicate else ''


def pandas_minmax(df, axis):
    return pd.DataFrame(minmax_scale(df, axis=axis), index=df.index, columns=df.columns)


def get_index_dict(df, level):
    d = defaultdict(set)
    level_ix = df.index.names.index(level)
    other_level_ix = 1 - level_ix
    for entry in df.index:
        d[entry[level_ix]].add(entry[other_level_ix])
    return d


def get_top_available(sorted_items: list, available_dict: dict, num_items_desired: int):
    return {
        key: set(itr.islice((item for item in sorted_items if item in available_items), num_items_desired))
        for key, available_items in available_dict.items()
    }


if __name__ == '__main__':
    from evaluation.helpers import get_data_block
    from time import time

    data_block, _, _, _ = get_data_block(0, 50, 0, 1000)
    start = time()
    m1 = pandas_minmax(data_block, axis=1)
    print(time() - start)
