import pandas as pd
from sklearn.preprocessing import minmax_scale
import ipdb
import itertools as itr


def pandas_minmax(df, axis):
    return pd.DataFrame(minmax_scale(df, axis=axis), index=df.index, columns=df.columns)


def get_index_dict(df, level):
    levels = pd.DataFrame({level: df.index.get_level_values(level=level) for level in df.index.names})
    other_level = (set(df.index.names) - {level}).pop()
    return levels.groupby(level)[other_level].apply(set).to_dict()


def get_top_available(sorted_items: list, available_dict: dict, num_items_desired: int):
    return {
        key: set(itr.islice((item for item in sorted_items if item in available_items), num_items_desired))
        for key, available_items in available_dict.items()
    }
