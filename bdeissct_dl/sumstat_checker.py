import os

import pandas as pd

from bdeissct_dl import TRAINING_PATH
from bdeissct_dl.training import get_test_data, get_X_columns
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest
from bdeissct_dl.bdeissct_model import DATA_TYPES, MODELS, BDCT


def check_sumstats(forest_sumstats, model_name):
    X, _ = get_test_data(dfs=[forest_sumstats])
    X_train, _ = get_test_data(paths=[os.path.join(TRAINING_PATH, model_name, f'{data_type}.csv.xz') for data_type in DATA_TYPES])

    feature_columns = get_X_columns(forest_sumstats.columns)

    df = pd.DataFrame(columns=['min', 'max', 'value', 'inside'])

    for i in range(len(feature_columns)):
        min_value = X_train[:, i].min()
        max_value = X_train[:, i].max()
        value = X[0, i]
        df.loc[feature_columns[i], :] = min_value, max_value, value, True
        if value - min_value < -1e-6 or max_value - value < -1e-6:
            df.loc[feature_columns[i], 'inside'] = False
            print(f'{feature_columns[i]:44s}\t{value :.6f}\tvs\t[{min_value :.6f}, {max_value :.6f}]')
    return df


def main():
    """
    Entry point for BDCT model finder with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Compare the summary statistics of a given forest "
                                            "to the training set used for a given model.")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--p', required=True, type=float, help='sampling probability')
    parser.add_argument('--model_name', choices=MODELS, default=BDCT, type=str,
                        help='phylodynamic model.')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    params = parser.parse_args()

    if params.p <= 0 or params.p > 1:
        raise ValueError('The sampling probability must be grater than 0 and not greater than 1.')

    forest = read_forest(params.nwk)
    print(f'Read a forest of {len(forest)} trees with {sum(len(_) for _ in forest)} tips in total')
    sumstat_df = forest2sumstat_df(forest, rho=params.p)

    check_sumstats(sumstat_df, params.model_name).to_csv(params.log, header=True)



if '__main__' == __name__:
    main()