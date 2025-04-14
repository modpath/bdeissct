import glob

import numpy as np
import re
from collections import Counter

from onnx.helper import get_node_attr_value
from treesumstats import FeatureManager

from bdei_model import F_I
from bdpn.bd_model import REPRODUCTIVE_NUMBER, INFECTIOUS_TIME, RHO
from bdpn.bdpn_model import REMOVAL_TIME_AFTER_NOTIFICATION
from bdpn.dl import MODELS, BD, BDCT1, BDCT2, BDCT2000, BDCT, MODEL_FINDER
from bdpn.dl.bdct_estimator import predict_parameters
from bdpn.dl.bdct_model_finder import predict_model
from bdpn.dl.tree_encoder import compute_extra_targets, STATS, scale, SCALING_FACTOR
from bdpn.dl.tree_encoder import forest2sumstat_df
from bdpn.tree_manager import read_forest, annotate_forest_with_time, TIME, rescale_forest_to_avg_brlen
from bdpn_model import UPSILON
from bdss_model import F_SS, X_SS
from dl.bdct_sumstat_checker import check_sumstats
from dl.tree_encoder import TARGET_AVG_BL

import pandas as pd

from model_distinguisher import ct_test


def remove_certain_leaves(tr, to_remove=lambda node: False):
    """
    Removes all the branches leading to leaves identified positively by to_remove function.
    :param tr: the tree of interest (ete3 Tree)
    :param to_remove: a method to check is a leaf should be removed.
    :return: void, modifies the initial tree.
    """

    tips = [tip for tip in tr if to_remove(tip)]
    for node in tips:
        if node.is_root():
            return None
        parent = node.up
        parent.remove_child(node)
        # If the parent node has only one child now, merge them.
        if len(parent.children) == 1:
            brother = parent.children[0]
            brother.dist += parent.dist
            if parent.is_root():
                brother.up = None
                tr = brother
            else:
                grandparent = parent.up
                grandparent.remove_child(parent)
                grandparent.add_child(brother)
    return tr

def _merge(l1, l2, key, max_size=np.inf):
    """
    Merges two sorted arrays
    :param l1: array 1
    :param l2: array 2
    :return: merged array
    """
    res = []
    i, j = 0, 0
    while len(res) < max_size and (i < len(l1) or j < len(l2)):
        if i == len(l1):
            res.extend(l2[j:min(len(l2), j + max_size - len(res))])
            break
        if j == len(l2):
            res.extend(l1[i: min(len(l1), i + max_size - len(res))])
            break
        if key(l1[i]) <= key(l2[j]):
            res.append(l1[i])
            i += 1
        else:
            res.append(l2[j])
            j += 1
    return res

def extract_clusters(tre, min_size, max_size):
    """
    Cuts the given tree into subtrees within a given size (s) range: min_size <= s <= max_size.
    The initial tree object is modified.

    :param max_size: minimal number of tips for a subtree (inclusive)
    :param min_size: maximal number of tips for a subtree (inclusive)
    :param tre: ete3.Tree
    :return: a generator of extracted subtrees
    """
    date_feature = TIME
    sorted_tips_feature = 'sorted-tips'
    taken_num_feature = 'taken'
    selection_strategy_feature = 'how'
    strategy_top = 'top'
    strategy_recursive = 'recurse'
    strategy_mixed = 'mixed_{}'

    def get_oldest_date(m):
        return getattr(getattr(m, sorted_tips_feature)[0], date_feature)

    def get_youngest_date(m):
        return getattr(getattr(m, sorted_tips_feature)[-1], date_feature)

    for n in tre.traverse('postorder'):
        n.add_feature(sorted_tips_feature,
                      [n] if n.is_leaf()
                      else _merge(*(getattr(_, sorted_tips_feature) for _ in n.children),
                                  key=lambda _: getattr(_, date_feature),
                                  max_size=max_size))
        n_size = len(n)

        if n_size < min_size:
            n.add_feature(taken_num_feature, 0)
        elif n_size <= max_size:
            n.add_feature(taken_num_feature, n_size)
            n.add_feature(selection_strategy_feature, strategy_top)
        else:
            taken = sum(getattr(_, taken_num_feature) for _ in n.children)
            how = strategy_recursive

            # if all the top leaves would come from just one of the children anyway,
            # the mixed solution will give the same result as recurse
            older_child, younger_child = sorted(n.children, key=get_oldest_date)
            if not (len(older_child) >= max_size and get_youngest_date(older_child) < get_oldest_date(younger_child)):
                tips = getattr(n, sorted_tips_feature)
                next_todo = list(n.children)
                for i in range(min_size, min(n_size, max_size) + 1):
                    date = getattr(tips[i - 1], date_feature)
                    size = i
                    todo = next_todo
                    next_todo = []
                    while todo:
                        m = todo.pop()

                        # if there is nothing to take here, no need to descend further
                        if not getattr(m, taken_num_feature):
                            continue

                        if getattr(getattr(m, sorted_tips_feature)[0], date_feature) <= date:
                            todo.extend(m.children)
                        else:
                            size += getattr(m, taken_num_feature)
                            next_todo.append(m)

                    if size > taken:
                        taken = size
                        how = strategy_mixed.format(i) if size > i else strategy_top
            n.add_feature(taken_num_feature, taken)
            n.add_feature(selection_strategy_feature, how)

    n_branches = 2 * len(tre) - 2
    n_subtrees = 0
    n_subtree_branches = 0
    for subtree in _dissect_tree(tre, min_size, max_size, date_feature,
                                 selection_strategy_feature, sorted_tips_feature, strategy_recursive, strategy_top):
        yield subtree
        n_subtrees += 1
        n_subtree_branches += 2 * len(subtree) - 2

    print(f'Picked {n_subtrees} subtrees covering {n_subtree_branches} out of {n_branches} branches ({100 * n_subtree_branches / n_branches:.1f}%).')


def _dissect_tree(tre, min_size, max_size, date_feature, selection_strategy_feature,
                 sorted_tips_feature, strategy_recursive, strategy_top):
    todo = [tre]
    while todo:
        n = todo.pop()
        if len(n) < min_size:
            continue
        how = getattr(n, selection_strategy_feature)
        if strategy_recursive == how:
            todo.extend(n.children)
            continue
        if strategy_top == how:
            n.detach()
            if len(n) <= max_size:
                yield n
                continue
            # tips should contain exactly max_size oldest tips
            tips = getattr(n, sorted_tips_feature)
            yield remove_certain_leaves(n, lambda _: _ not in tips)
            continue
        # strategy mixed in action
        i = int(how[6:])
        date = getattr(getattr(n, sorted_tips_feature)[i - 1], date_feature)
        child_todo = list(n.children)
        while child_todo:
            m = child_todo.pop()
            if getattr(getattr(m, sorted_tips_feature)[0], date_feature) <= date:
                child_todo.extend(m.children)
            else:
                parent = m.up
                todo.append(m.detach())
                if parent.up:
                    for c in parent.children:
                        parent.up.add_child(c, dist=c.dist + parent.dist)
                    parent.up.remove_child(parent)
        yield n.detach()


def extract_subtrees():
    for i in range(10):

        NWK = f'/home/azhukova/projects/bdpn/hiv_b_uk/data/timetree.{i}.nexus'
        forest = read_forest(NWK)

        root_date = 0
        with open(f'/home/azhukova/projects/bdpn/hiv_b_uk/data/timetree.{i}.log', 'r') as f:
            for line in f.readlines():
                if 'tMRCA' in line:
                    root_date = float(re.findall(r'tMRCA\s+(\d\d\d\d[.\d]+)\s*,', line)[0])
                    print(root_date)

        annotate_forest_with_time(forest, start_times=[root_date])
        # forest = [remove_certain_leaves(forest[0], lambda t: getattr(t, TIME) >= 2016)]
        todo = forest
        k = 0
        total = 0
        while todo:
            n = todo.pop()
            if len(n) < 200:
                continue
            if getattr(n, TIME) < 1998: #1996.0846994535518:
                todo.extend(n.children)
                continue
            for clu in extract_clusters(n, 200, 2000):
                clu.write(outfile=f'/home/azhukova/projects/bdpn/hiv_b_uk/data/subtree.uk.{i}.{k}.nwk')
                print(len(clu), getattr(clu, TIME), max(getattr(_, TIME) for _ in clu))
                k += 1
                total += len(clu)
        print(total)
# extract_subtrees()


nwks = glob.glob('/home/azhukova/projects/bdpn/hiv_b_uk/data/timesubrax.*.nexus')

for nwk in nwks:
    forest = read_forest(nwk)
    sumstats = forest2sumstat_df(forest, rho=0.58)
    model_df = predict_model(sumstats)
    print(model_df.loc[0, :])
    for model in MODELS:
        print(model)
        check_sumstats(sumstats, model_name=model)

# model_counter = Counter()
# print(f'Detected {len(nwks)} subtrees.')
# for nwk in nwks:
#     forest = read_forest(nwk)
#     sumstat_df = forest2sumstat_df(forest, rho=0.58)
#     # check_sumstats(sumstat_df, BDCT2)
#     model_df = predict_model(sumstat_df)
#     # print(model_df.loc[0, :])
#     for model, w in model_df.to_dict(orient='list').items():
#         model_counter[model] += w[0] / len(nwks)
#
#     # best_i = np.argmax(model_df.loc[0, :])
#     # model = MODELS[best_i]
#     # print(f'The best model is {model} with probability {model_df.iloc[0, best_i]}')
#
# print(model_counter)



stats = []
weights = []
ct = []
for nwk in nwks:
    forest = read_forest(nwk)
    ct.append(ct_test(forest)[0])
    weights.append(2 * len(forest[0]) - 1)
    scaling_factor = rescale_forest_to_avg_brlen(forest, target_avg_length=TARGET_AVG_BL)

    kwargs = {SCALING_FACTOR: scaling_factor, RHO: 0.58}
    scale(kwargs, scaling_factor)

    stats.append(list(FeatureManager.compute_features(forest, *STATS, **kwargs)))


weights = np.array(weights, dtype=float)
weights /= weights.sum()


print(f'ct:\t{np.average(ct, weights=weights):.2f}\t[{np.quantile(ct, 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(ct, 0.975, weights=weights, method='inverted_cdf'):.2f}]', min(ct), max(ct))

sumstat_df = pd.DataFrame.from_records(stats, columns=STATS)
model_df = predict_model(sumstat_df)
model_df.loc['avg', :] = (model_df.to_numpy(dtype=float) * np.reshape(weights.T, (len(nwks), 1))).sum(axis=0)
print(model_df.loc['avg', :])


Y_pred = predict_parameters(sumstat_df, MODEL_FINDER, ci=True)
compute_extra_targets(Y_pred)
for col in Y_pred.columns:
    if REMOVAL_TIME_AFTER_NOTIFICATION in col:
        Y_pred[col] *= 12
    # print(Y_pred.loc[0, :].apply(lambda _: f'{_:.2f}'))

# print(Y_pred)

# Y_pred.loc['avg', :] = Y_pred.apply(lambda x: (np.asarray(x) * weights.T).sum(axis=1))
# print(Y_pred.loc['avg', :])



# prts = np.array(prts) * 365
#
# print(len(weights), len(prts), len(ups))

print(f'R0:\t{np.average(Y_pred[REPRODUCTIVE_NUMBER].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[REPRODUCTIVE_NUMBER], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[REPRODUCTIVE_NUMBER], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
print(f'd:\t{np.average(Y_pred[INFECTIOUS_TIME].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[INFECTIOUS_TIME], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[INFECTIOUS_TIME], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
print(f'prt:\t{np.average(Y_pred[REMOVAL_TIME_AFTER_NOTIFICATION].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[REMOVAL_TIME_AFTER_NOTIFICATION], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[REMOVAL_TIME_AFTER_NOTIFICATION], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
print(f'ups:\t{np.average(Y_pred[UPSILON].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[UPSILON], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[UPSILON], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
print(f'f_i:\t{np.average(Y_pred[F_I].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[F_I], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[F_I], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
print(f'f_ss:\t{np.average(Y_pred[F_SS].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[F_SS], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[F_SS], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
print(f'x_ss:\t{np.average(Y_pred[X_SS].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[X_SS], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[X_SS], 0.975, weights=weights, method='inverted_cdf'):.2f}]')