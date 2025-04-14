import glob

import numpy as np

from bdpn.dl import MODELS, BD, BDCT1, BDCT2, BDCT2000, BDCT, MODEL_FINDER
from bdpn.dl.bdct_estimator import predict_parameters
from bdpn.dl.bdct_model_finder import predict_model
from bdpn.dl.tree_encoder import compute_extra_targets, STATS, scale, SCALING_FACTOR
from bdpn.dl.tree_encoder import forest2sumstat_df
from pastml.tree import read_forest, annotate_dates, remove_certain_leaves

from model_distinguisher import ct_test



def extract_subtrees():
    i = 9
    NWK = f'/home/azhukova/projects/bdei_main/ebola/data/pastml/{i}/named.tree_timetree.{i}.nwk'
    forest = read_forest(NWK, columns=['country'])
    annotate_dates(forest)
    todo = [forest[0]]
    while todo:
        node = todo.pop()
        if len(node) < 200:
            continue
        if 'SLE' not in getattr(node, 'country') or node.dist < 2 / 18996:
            todo.extend(node.children)
            continue
    # if True:
    #     node = forest[0] # 2014.5780821917808
        tree = remove_certain_leaves(node,
                                     lambda t: getattr(t, 'date') > 2014.65 or 'SLE' not in getattr(t, 'country'))
        tree.write(outfile='/home/azhukova/Downloads/subtree.nwk')
        print(len(tree))


extract_subtrees()



tree = read_forest('/home/azhukova/Downloads/subtree.nwk')[0]
print(ct_test([tree]))
sumstat_df = forest2sumstat_df([tree], rho=0.06)
model_df = predict_model(sumstat_df)
print(model_df.loc[0, :])

Y_pred = predict_parameters(sumstat_df, MODEL_FINDER, ci=True)
compute_extra_targets(Y_pred)
for col in Y_pred.columns:
    if 'time' in col:
        Y_pred[col] *= 365
Y_pred.loc[0, 'incubation time'] = Y_pred.loc[0, 'infectious time'] * Y_pred.loc[0, 'f_inc'] / (1 - Y_pred.loc[0, 'f_inc'])
print(Y_pred.loc[0, ['R0', 'infectious time', 'incubation time', 'x', 'f_ss', 'upsilon', 'removal time after notification']].apply(lambda _: f'{_:.2f}'))

# print(Y_pred)

# Y_pred.loc['avg', :] = Y_pred.apply(lambda x: (np.asarray(x) * weights.T).sum(axis=1))
# print(Y_pred.loc['avg', :])



# prts = np.array(prts) * 365
#
# print(len(weights), len(prts), len(ups))

# print(f'R0:\t{np.average(Y_pred[REPRODUCTIVE_NUMBER].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[REPRODUCTIVE_NUMBER], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[REPRODUCTIVE_NUMBER], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
# print(f'd:\t{np.average(Y_pred[INFECTIOUS_TIME].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[INFECTIOUS_TIME], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[INFECTIOUS_TIME], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
# print(f'prt:\t{np.average(Y_pred[REMOVAL_TIME_AFTER_NOTIFICATION].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[REMOVAL_TIME_AFTER_NOTIFICATION], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[REMOVAL_TIME_AFTER_NOTIFICATION], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
# print(f'ups:\t{np.average(Y_pred[UPSILON].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[UPSILON], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[UPSILON], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
# print(f'f_i:\t{np.average(Y_pred[F_I].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[F_I], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[F_I], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
# print(f'f_ss:\t{np.average(Y_pred[F_SS].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[F_SS], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[F_SS], 0.975, weights=weights, method='inverted_cdf'):.2f}]')
# print(f'x_ss:\t{np.average(Y_pred[X_SS].to_numpy(), weights=weights):.2f}\t[{np.quantile(Y_pred[X_SS], 0.025, weights=weights, method='inverted_cdf'):.2f}, {np.quantile(Y_pred[X_SS], 0.975, weights=weights, method='inverted_cdf'):.2f}]')