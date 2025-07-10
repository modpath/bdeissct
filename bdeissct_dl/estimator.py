import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.bdeissct_model import F_E, RHO, UPSILON, MODEL2TARGET_COLUMNS, BD, BDEI, BDSS, BDEISS, MODELS, \
    MODEL_FINDER, F_S, X_S, X_C, PI_I, PI_IC, PI_EC, PI_S, PI_E, PI_SC, LA, PSI, UPSILON, PIS
from bdeissct_dl.model_serializer import load_model_keras, load_scaler_numpy
from bdeissct_dl.training import get_test_data
from bdeissct_dl.tree_encoder import forest2sumstat_df, scale_back
from bdeissct_dl.tree_manager import read_forest

def predict_parameters(forest_sumstats, model_name=MODEL_FINDER, model_path=MODEL_PATH):
    n_forests = len(forest_sumstats)
    n_models = len(MODELS)

    if MODEL_FINDER == model_name:
        import bdeissct_dl.training_model_finder
        X = bdeissct_dl.training_model_finder.get_test_data(df=forest_sumstats)
        model_weights = load_model_keras(model_path.format(model_name)).predict(X)
    else:
        model_weights = np.zeros((n_forests, n_models), dtype=float)
        model_weights[:, MODELS.index(model_name)] = 1



    X, SF = get_test_data(dfs=[forest_sumstats], scaler_x=None)

    results = []

    # result = pd.DataFrame(index=np.arange(X.shape[0]))

    model_ids = [i for i in range(n_models) if not np.all(model_weights[:, i] == 0)]
    for model_id in model_ids:
        model_name = MODELS[model_id]

        model_path = model_path.format(model_name)

        X_cur, SF_cur = np.array(X), np.array(SF)

        scaler_x = load_scaler_numpy(model_path, suffix='x')
        if scaler_x:
            X_cur = scaler_x.transform(X_cur)

        model = load_model_keras(model_path)
        Y_pred = model.predict(X_cur)

        target_columns = MODEL2TARGET_COLUMNS[model_name]
        Y_pred[PI_E] = Y_pred[PIS][:, 0]
        Y_pred[PI_I] = Y_pred[PIS][:, 1]
        Y_pred[PI_S] = Y_pred[PIS][:, 2]
        Y_pred[PI_EC] = Y_pred[PIS][:, 3]
        Y_pred[PI_IC] = Y_pred[PIS][:, 4]
        Y_pred[PI_SC] = Y_pred[PIS][:, 5]

        del Y_pred[PIS]

        for col in target_columns:
            if len(Y_pred[col].shape) == 2 and Y_pred[col].shape[1] == 1:
                Y_pred[col] = Y_pred[col].squeeze(axis=1)

        scale_back(Y_pred, SF_cur)

        results.append(pd.DataFrame.from_dict(Y_pred, orient='columns'))

    if len(model_ids) == 1:
        result = results[0]
    else:
        # if any('CT' in MODELS[_] for _ in model_ids):
        #     ups_cols = [(f'{UPSILON}_{q * 100:.1f}' if q != 0.5 else UPSILON) for q in QUANTILES]
        #     x_c_cols = [(f'{X_C}_{q * 100:.1f}' if q != 0.5 else X_C) for q in QUANTILES]
        #     bdeiss_ids = {_[0] for _ in enumerate(model_ids) if MODELS[_[1]] in (BD, BDEI, BDSS, BDEISS)}
        #     for idx in bdeiss_ids:
        #         results[idx].loc[:, ups_cols] = np.zeros((n_forests, len(ups_cols)), dtype=float)
        #         results[idx].loc[:, x_c_cols] = np.ones((n_forests, len(ups_cols)), dtype=float)
        # bdei_ids = {_[0] for _ in enumerate(model_ids) if 'EI' in MODELS[_[1]]}
        # if bdei_ids and len(bdei_ids) < len(model_ids):
        #     f_cols = [(f'{F_E}_{q * 100:.1f}' if q != 0.5 else F_E) for q in QUANTILES]
        #     for idx in range(len(model_ids)):
        #         if not idx in bdei_ids:
        #             results[idx].loc[:, f_cols] = np.zeros((n_forests, len(f_cols)), dtype=float)
        # bdss_ids = {_[0] for _ in enumerate(model_ids) if 'SS' in MODELS[_[1]]}
        # if bdss_ids and len(bdss_ids) < len(model_ids):
        #     f_cols = [(f'{F_S}_{q * 100:.1f}' if q != 0.5 else F_S) for q in QUANTILES]
        #     x_cols = [(f'{X_S}_{q * 100:.1f}' if q != 0.5 else X_S) for q in QUANTILES]
        #     for idx in range(len(model_ids)):
        #         if not idx in bdss_ids:
        #             results[idx].loc[:, f_cols] = np.zeros((n_forests, len(f_cols)), dtype=float)
        #             results[idx].loc[:, x_cols] = np.ones((n_forests, len(f_cols)), dtype=float)

        columns = results[0].columns
        result = pd.DataFrame(index=forest_sumstats.index)
        for col in columns:
            predictions = np.array([res[col].to_numpy(dtype=float, na_value=0) for res in results]).T
            weights = model_weights[:, model_ids]
            result[col] = np.average(predictions, weights=weights, axis=1)

    return result


def main():
    """
    Entry point for tree parameter estimation with a BDCT model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimate BD(EI)(SS)(CT) model parameters.")
    parser.add_argument('--model_name', choices=MODELS + (MODEL_FINDER,), default=BD, type=str,
                        help=f'BDEISSCT model flavour. If {MODEL_FINDER} is specified, '
                             f'model finder will be used to pick the model.')
    parser.add_argument('--model_path_pattern', default=MODEL_PATH,
                        help='By default our pretrained BD(EI)(SS)(CT) models are used, '
                             'but it is possible to specify a pattern of a path to a custom folder here, '
                             'containing files "ffnn.keras" (with the model), '
                             'and scaler-related files to rescale the input data X, and the output Y: '
                             'for X: "data_scalerx_mean.npy", "data_scalerx_scale.npy", "data_scalerx_var.npy" '
                             '(unpickled numpy-saved arrays), '
                             'and "data_scalerx_n_samples_seen.txt" '
                             'a text file containing the number of examples in the training set). '
                             'For Y the file names are the same, just x replaced by y, e.g., "data_scalery_mean.npy". '
                             'The path pattern should contain a part "{}", which will be replaced by the model name, '
                             'e.g., "/home/user/models/{}/", which for a model BD will point to "/home/user/models/BD/"')
    parser.add_argument('--p', default=0, type=float, help='sampling probability')
    parser.add_argument('--log', default='/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/500_1000/BD/trees.estimates_BD', type=str, help="output log file")
    parser.add_argument('--nwk', default=None, type=str, help="input tree file")
    parser.add_argument('--sumstats', default=None, type=str, help="input tree file(s) encoded as sumstats")
    parser.add_argument('--ci', action='store_true', help="calculate CIs")
    params = parser.parse_args()

    if not params.sumstats:
        if params.p <= 0 or params.p > 1:
            raise ValueError('The sampling probability must be grater than 0 and not greater than 1.')

        forest = read_forest(params.nwk)
        print(f'Read a forest of {len(forest)} trees with {sum(len(_) for _ in forest)} tips in total')
        forest_df = forest2sumstat_df(forest, rho=params.p)
    else:
        forest_df = pd.read_csv(params.sumstats)
    predict_parameters(forest_df, model_name=params.model_name, model_path=params.model_path_pattern)\
        .to_csv(params.log, header=True)


if '__main__' == __name__:
    main()