import os

import numpy as np
import pandas as pd

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.bdeissct_model import (MODEL2TARGET_COLUMNS, BD, INCUBATION_FRACTION, F_S, X_S, UPSILON, X_C,
                                        REPRODUCTIVE_NUMBER, INFECTION_DURATION)
from bdeissct_dl.model_serializer import load_model_keras, load_scaler_numpy
from bdeissct_dl.training import get_test_data
from bdeissct_dl.tree_encoder import forest2sumstat_df, scale_back
from bdeissct_dl.tree_manager import read_forest


def predict_parameters_by_column(forest_sumstats, model_name=BD, model_path=MODEL_PATH):
    scaler_x = load_scaler_numpy(model_path, suffix=f'{model_name}.x')
    scaler_y = load_scaler_numpy(model_path, suffix=f'{model_name}.y')
    X, SF = get_test_data(dfs=[forest_sumstats], scaler_x=scaler_x)

    target_columns = list(MODEL2TARGET_COLUMNS[model_name])

    result = None
    for col in target_columns:
        model = load_model_keras(model_path, f'{model_name}.{col}')
        Y_pred = model.predict(X)

        if len(Y_pred[col].shape) == 2 and Y_pred[col].shape[1] == 1:
            Y_pred[col] = Y_pred[col].squeeze(axis=1)

        res_df = pd.DataFrame.from_dict(Y_pred, orient='columns')
        result = result.join(res_df, how='outer') if result is not None else res_df

    if scaler_y is not None:
        Y_pred = result[target_columns].to_numpy(dtype=float, na_value=0)
        Y_pred = scaler_y.inverse_transform(Y_pred)
        result = pd.DataFrame(Y_pred, columns=target_columns)
    scale_back(result, SF)

    return result

def predict_parameters(forest_sumstats, model_name=BD, model_path=MODEL_PATH):
    scaler_x = load_scaler_numpy(model_path, suffix=f'{model_name}.x')
    scaler_y = load_scaler_numpy(model_path, suffix=f'{model_name}.y')
    X, SF = get_test_data(dfs=[forest_sumstats], scaler_x=scaler_x)

    model = load_model_keras(model_path, f'{model_name}')
    Y_pred = model.predict(X)
    target_columns = MODEL2TARGET_COLUMNS[model_name]
    for col in target_columns:
        if len(Y_pred[col].shape) == 2 and Y_pred[col].shape[1] == 1:
            Y_pred[col] = Y_pred[col].squeeze(axis=1)
            # if Y_pred[col].shape[0] == 1:
            #     Y_pred[col] = Y_pred[col][0]
    Y_pred = np.column_stack([Y_pred[col] for col in target_columns])
    if scaler_y is not None:
        Y_pred = scaler_y.inverse_transform(Y_pred)
    Y_pred = pd.DataFrame(Y_pred, columns=target_columns)
    scale_back(Y_pred, SF)
    return Y_pred


def main():
    """
    Entry point for tree parameter estimation with a BDCT model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimate BD(EI)(SS)(CT) model parameters.")
    parser.add_argument('--model_name', default=BD, type=str,
                        help=f'BDEISSCT model flavour')
    parser.add_argument('--model_path', default=MODEL_PATH,
                        help='By default our pretrained BD(EI)(SS)(CT) models are used, '
                             'but it is possible to specify a path to a custom folder here, '
                             'containing files "<model_name>.keras" (with the model), '
                             'and scaler-related files to rescale the input data X and output data Y: '
                             '(for X) "data_scaler<model_name>.x_mean.npy", "data_scaler<model_name>.x_scale.npy", '
                             '"data_scaler<model_name>.x_var.npy" '
                             '(unpickled numpy-saved arrays), '
                             'and "data_scaler<model_name>.x_n_samples_seen.txt" '
                             'a text file containing the number of examples in the training set. '
                             'For the Y scaler replace x with y in the filenames above.'
                        )
    parser.add_argument('--p', default=0, type=float, help='sampling probability')
    parser.add_argument('--log', default=None, type=str, help="output log file")
    parser.add_argument('--nwk', default=None, type=str, help="input tree file")
    parser.add_argument('--sumstats', default=None, type=str, help="input tree file(s) encoded as sumstats")
    parser.add_argument('--ci', action='store_true', help="calculate CIs")
    params = parser.parse_args()

    if not params.sumstats:
        if params.p <= 0 or params.p > 1:
            raise ValueError('The sampling probability must be between 0 (exclusive) and 1 (inclusive).')

        forest = read_forest(params.nwk)
        print(f'Read a tree with {sum(len(_) for _ in forest)} tips.')
        forest_df = forest2sumstat_df(forest, rho=params.p)
    else:
        forest_df = pd.read_csv(params.sumstats)
    if os.path.exists(os.path.join(params.model_path, f'{params.model_name}.keras')):
        result = predict_parameters(forest_df, model_name=params.model_name, model_path=params.model_path)
    else:
        result = predict_parameters_by_column(forest_df, model_name=params.model_name, model_path=params.model_path)

    enforce_BDEISSCT_bounds(result)

    result.to_csv(params.log, header=True)


def enforce_BDEISSCT_bounds(result):
    for col in result.columns:
        result[col] = np.maximum(result[col], 0 if col not in {X_C, X_S} else 1)
        if col in {INCUBATION_FRACTION, UPSILON}:
            result[col] = np.minimum(result[col], 1)
        elif col in {F_S}:
            result[col] = np.minimum(result[col], 0.5)


if '__main__' == __name__:
    main()