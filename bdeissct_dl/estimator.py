import os

import numpy as np
import pandas as pd

from bdeissct_dl.bdeissct_model import (MODEL2TARGET_COLUMNS, BD, INCUBATION_FRACTION, F_S, X_S, UPSILON, X_C)
from bdeissct_dl.model_serializer import load_model_keras, load_scaler_numpy, get_model_dir, RANDOM_SEED
from bdeissct_dl.training import get_test_data
from bdeissct_dl.tree_encoder import forest2sumstat_df, scale_back
from bdeissct_dl.tree_manager import read_forest


def predict_parameters(forest_sumstats, model_name=BD, model_path=None, seed=RANDOM_SEED):
    if model_path is None:
        min_tips = int(forest_sumstats['n_tips'].min())
        max_tips = int(forest_sumstats['n_tips'].max())
        model_path = get_model_dir(min_tips, max_tips)

    scaler_x = load_scaler_numpy(model_path, suffix=f'{model_name}.x')
    scaler_y = load_scaler_numpy(model_path, suffix=f'{model_name}.y')
    X, SF = get_test_data(dfs=[forest_sumstats], scaler_x=scaler_x)


    target_columns = MODEL2TARGET_COLUMNS[model_name]
    seed = '' if seed <= 0 else f'.{seed}'
    model = load_model_keras(os.path.join(model_path, f'{model_name}{seed}.keras'))
    Y_pred = model.predict(X)
    quantiles=False
    for col in target_columns:
        if len(Y_pred[col].shape) == 2:
            if Y_pred[col].shape[1] == 1:
                Y_pred[col] = Y_pred[col].squeeze(axis=1)
            else:
                quantiles = True
    if not quantiles:
        Y_pred = np.column_stack([Y_pred[col] for col in target_columns])
        if scaler_y is not None:
            Y_pred = scaler_y.inverse_transform(Y_pred)
        Y_pred = pd.DataFrame(Y_pred, columns=target_columns)
    else:
        Y_pred_min = np.column_stack([Y_pred[col][:, 0] for col in target_columns])
        Y_pred_median = np.column_stack([Y_pred[col][:, 1] for col in target_columns])
        Y_pred_max = np.column_stack([Y_pred[col][:, 2] for col in target_columns])
        if scaler_y is not None:
            Y_pred_min = scaler_y.inverse_transform(Y_pred_min)
            Y_pred_median = scaler_y.inverse_transform(Y_pred_median)
            Y_pred_max = scaler_y.inverse_transform(Y_pred_max)
        Y_pred = pd.concat([pd.DataFrame(Y_pred_median, columns=target_columns),
                            pd.DataFrame(Y_pred_min, columns=[f'{col}_lower' for col in target_columns]),
                            pd.DataFrame(Y_pred_max, columns=[f'{col}_upper' for col in target_columns])], axis=1)

    scale_back(Y_pred, SF)

    enforce_BDEISSCT_bounds(Y_pred)
    return Y_pred


def main():
    """
    Entry point for tree parameter estimation with a BD(EI)(SS)(-CT) model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimate BD(EI)(SS)(CT) model parameters.")
    parser.add_argument('--model_name', default=BD, type=str,
                        help=f'BDEISSCT model flavour')
    parser.add_argument('--model_path', default=None, type=str,
                        help=f'By default our pretrained BD(EI)(SS)(CT) models are used (with seed {RANDOM_SEED}), '
                             'but it is possible to specify a path to a custom folder here, '
                             'containing scaler-related files to rescale the input data X and output data Y and the model file(s). '
                             'For the X scaler the files should be named '
                             '"data_scaler<model_name>.x_mean.npy", "data_scaler<model_name>.x_scale.npy", '
                             '"data_scaler<model_name>.x_var.npy" '
                             '(unpickled numpy-saved arrays), '
                             'and "data_scaler<model_name>.x_n_samples_seen.txt" '
                             '(a text file containing the number of examples in the training set). '
                             'For the Y scaler replace x with y in the filenames above. '
                             'The model file can be either "<model_name>.keras" (containing a model with a random seed value set randomly), '
                             'or "<model_name>.<seed>.keras" '
                             '(containing a model with a random seed value set to seed specified as --seed).'
                        )
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='if a non-negative number is given, '
                                                             'it will be searched for in the model name.')
    parser.add_argument('--p', default=-1, type=float, help='sampling probability')
    parser.add_argument('--log', default=None, type=str, help="output log file")
    parser.add_argument('--nwk', default=None, type=str, help="input tree file")
    parser.add_argument('--sumstats', default=None, type=str, help="input tree file(s) encoded as sumstats")
    params = parser.parse_args()

    estimate_main(**vars(params))


def estimate_main(log, model_name, sumstats=None, p=-1, nwk=None, model_path=None, seed=RANDOM_SEED):
    if not sumstats:
        if p <= 0 or p > 1:
            raise ValueError('The sampling probability must be between 0 (exclusive) and 1 (inclusive).')

        forest = read_forest(nwk)
        print(f'Read a tree with {sum(len(_) for _ in forest)} tips.')
        forest_df = forest2sumstat_df(forest, rho=p)
    else:
        forest_df = pd.read_csv(sumstats)

    result = predict_parameters(forest_df, model_name=model_name, model_path=model_path, seed=seed)
    result.to_csv(log, header=True)


def enforce_BDEISSCT_bounds(result):
    for col in result.columns:
        if col.endswith('_std'):
            continue
        result[col] = np.maximum(result[col], 0 if not (col.startswith(X_C) or col.startswith(X_S)) else 1)
        if col.startswith(INCUBATION_FRACTION) or col.startswith(UPSILON):
            result[col] = np.minimum(result[col], 1)
        elif col.startswith(F_S):
            result[col] = np.minimum(result[col], 0.5)


if '__main__' == __name__:
    main()