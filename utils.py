import pandas as pd
from sklearn.preprocessing import minmax_scale
import ipdb
import itertools as itr
from collections import defaultdict, Counter


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


class NoContextsWithTargetAction(Exception):
    pass


class NoDonorActionsWithTargetContext(Exception):
    pass


def find_donors(
        sorted_donors,
        contexts2available_actions,
        target_context,
        target_action,
        num_desired_donors=None
):
    donor_actions_with_target_context = [
        donor for donor in sorted_donors
        if donor in contexts2available_actions[target_context]
    ]
    # print(f"num donor actions: {len(donor_actions_with_target_context)}")
    if len(donor_actions_with_target_context) == 0:
        raise NoDonorActionsWithTargetContext

    # start with training data which has the target_donor_ix available
    current_training_contexts = [
        context for context, available_actions in contexts2available_actions.items()
        if target_action in available_actions
    ]
    # print(f"num training contexts: {len(current_training_contexts)}")
    if len(current_training_contexts) == 0:
        raise NoContextsWithTargetAction

    donor_actions = set()
    while True:
        if len(donor_actions_with_target_context) == 0:
            break
        next_donor_action = donor_actions_with_target_context.pop(0)

        # stop adding donors if it would make the number of training dimensions zero
        new_training_contexts = [
            train_ix for train_ix in current_training_contexts
            if next_donor_action in contexts2available_actions[train_ix]
        ]
        if len(new_training_contexts) == 0:
            break

        # stop adding donors if we've reached the desired number of donors, and adding another
        # donor would decrease the number of training dimensions
        reached_num_desired = len(donor_actions) == num_desired_donors
        if reached_num_desired and len(new_training_contexts) < len(current_training_contexts):
            break

        # otherwise, it's fine to add another donor
        current_training_contexts = new_training_contexts
        donor_actions.add(next_donor_action)

    assert len(current_training_contexts) > 0
    assert len(donor_actions) > 0
    return current_training_contexts, donor_actions


class DonorFinder:
    def __init__(self, df, regression_dim="intervention", context_dim="unit"):
        df = df.sort_index(level=[regression_dim, context_dim])

        counter = Counter(df.index.get_level_values(regression_dim))
        self.contexts2available_actions = get_index_dict(df, context_dim)
        self.sorted_donors = [donor for donor, _ in counter.most_common()]

    def get_donors(self, target_context, target_action, num_desired_donors=None):
        training_contexts, donor_actions = find_donors(
            self.sorted_donors,
            self.contexts2available_actions,
            target_context,
            target_action,
            num_desired_donors
        )
        return training_contexts, donor_actions


if __name__ == '__main__':
    from evaluation.helpers import get_data_block
    from time import time

    data_block, _, _, _ = get_data_block(0, 50, 0, 1000)
    start = time()
    m1 = pandas_minmax(data_block, axis=1)
    print(time() - start)
