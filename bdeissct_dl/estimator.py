import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.dl_model import QUANTILES
from bdeissct_dl.bdeissct_model import F_E, RHO, UPSILON, MODEL2TARGET_COLUMNS, BD, BDEI, BDSS, BDEISS, MODELS, \
    MODEL_FINDER, F_S, X_S, X_C
from bdeissct_dl.model_serializer import load_model_keras, load_scaler_numpy
from bdeissct_dl.training import get_test_data
from bdeissct_dl.tree_encoder import forest2sumstat_df, scale_back_array
from bdeissct_dl.tree_manager import read_forest


def estimate_cis(forest_sumstats, model_name, Y_pred):
    target_columns = list(MODEL2TARGET_COLUMNS[model_name])
    train_sumstats = pd.read_csv(os.path.join(TRAINING_PATH, f'{model_name}.csv.xz'))[target_columns + [RHO, 'n_trees', 'n_tips']]
    train_results = pd.read_csv(os.path.join(TRAINING_PATH, f'{model_name}.estimates_{model_name}.csv.xz'), index_col=0)[target_columns]
    Y_pred = Y_pred[target_columns]

    scaler_y = StandardScaler()
    train_results = scaler_y.fit_transform(train_results.to_numpy(dtype=float, na_value=0))
    Y_pred = scaler_y.transform(Y_pred.to_numpy(dtype=float, na_value=0))

    Y_pred = pd.DataFrame(Y_pred, columns=target_columns, index=forest_sumstats.index)
    train_results = pd.DataFrame(train_results, columns=target_columns)

    n_train_examples = len(train_sumstats)
    tip_threshold = 200_000 if n_train_examples > 200_000 else 40_000
    tree_threshold = 100_000 if n_train_examples > 200_000 else 20_000
    rho_threshold = 10_000
    param_threshold = 1000


    for index in forest_sumstats.index:
        rho, n_trees, n_tips = forest_sumstats.loc[index, [RHO, 'n_trees', 'n_tips']]
        cur_sumstats = train_sumstats
        cur_results = train_results
        # print('tips', n_tips, np.percentile(train_sumstats['n_tips'], [0, 5, 50, 95, 100]))
        cur_ids = cur_sumstats.iloc[np.abs(cur_sumstats[RHO] - rho).argsort(), :].index[:tip_threshold]
        cur_sumstats = cur_sumstats.loc[cur_ids, :]
        cur_results = cur_results.loc[cur_ids, :]
        # print('tips', n_tips, np.percentile(cur_sumstats['n_tips'], [0, 5, 50, 95, 100]))
        # print('trees', n_trees, np.percentile(cur_sumstats['n_trees'], [0, 5, 50, 95, 100]))
        cur_ids = cur_sumstats.iloc[np.abs(cur_sumstats['n_trees'] - n_trees).argsort(), :].index[:tree_threshold]
        cur_sumstats = cur_sumstats.loc[cur_ids, :]
        cur_results = cur_results.loc[cur_ids, :]
        # print('trees', n_trees, np.percentile(cur_sumstats['n_trees'], [0, 5, 50, 95, 100]))
        # print(RHO, rho, np.percentile(cur_sumstats[RHO], [0, 5, 50, 95, 100]))
        cur_ids = cur_sumstats.iloc[np.abs(cur_sumstats['n_tips'] - n_tips).argsort(), :].index[:rho_threshold]
        cur_sumstats = cur_sumstats.loc[cur_ids, :]
        cur_results = cur_results.loc[cur_ids, :]
        # print(RHO, rho, np.percentile(cur_sumstats[RHO], [0, 5, 50, 95, 100]))

        loss = np.abs(cur_sumstats[target_columns] - Y_pred.loc[index, target_columns]).sum(axis=1)
        cur_ids = cur_sumstats.iloc[loss.argsort(), :].index[:param_threshold]
        cur_sumstats = cur_sumstats.loc[cur_ids, :]
        cur_results = cur_results.loc[cur_ids, :]


        for col in target_columns:
            val = Y_pred.loc[index, col]
            # # print(col, val, np.percentile(cur_sumstats[col], [0, 5, 50, 95, 100]), np.percentile(cur_results[col], [0, 5, 50, 95, 100]))
            # par_ids = cur_sumstats.iloc[np.abs(cur_sumstats[col] - val).argsort(), :].index[:param_threshold]
            # true_vals = cur_sumstats.loc[par_ids, col]
            # pred_vals = cur_results.loc[par_ids, col]
            # # print(col, val, np.percentile(true_vals, [0, 5, 50, 95, 100]), np.percentile(pred_vals, [0, 5, 50, 95, 100]))
            # errors = pred_vals - true_vals
            errors = cur_results[col] - cur_sumstats[col]
            dist = val + errors - np.median(errors)
            # dist = np.maximum(dist, 0)
            # if UPSILON == col:
            #     dist = np.minimum(dist, 1)
            Y_pred.loc[index, f'{col}_2.5'] = np.percentile(dist, 2.5)
            Y_pred.loc[index, f'{col}_97.5'] = np.percentile(dist, 97.5)
    for suffix in ('', '_2.5', '_97.5'):
        Y_pred[[f'{col}{suffix}' for col in target_columns]] = \
            scaler_y.inverse_transform(Y_pred[[f'{col}{suffix}' for col in target_columns]].to_numpy(dtype=float, na_value=0))


    Y_pred = np.maximum(Y_pred, 0)
    ups_cols = [col for col in Y_pred.columns if UPSILON in col]
    if ups_cols:
        Y_pred[ups_cols] = np.minimum(Y_pred[ups_cols], 1)
    return Y_pred


def predict_parameters(forest_sumstats, model_name=MODEL_FINDER, ci=False, model_path=MODEL_PATH):
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
        scaler_y = load_scaler_numpy(model_path, suffix='y')
        if scaler_y:
            n_quant = len(QUANTILES)
            for i in range(n_quant):
                Y_pred[:, i::n_quant] = scaler_y.inverse_transform(Y_pred[:, i::n_quant])

        target_columns = MODEL2TARGET_COLUMNS[model_name]
        tc_cis = tuple((f'{_}_{q * 100:.1f}' if q != 0.5 else _) for _ in target_columns
                       for q in QUANTILES)
        Y_pred = np.maximum(Y_pred, 0)
        scale_back_array(Y_pred, SF_cur, tc_cis)
        results.append(pd.DataFrame(Y_pred, columns=tc_cis))

    if len(model_ids) == 1:
        result = results[0]
    else:
        if any('CT' in MODELS[_] for _ in model_ids):
            ups_cols = [(f'{UPSILON}_{q * 100:.1f}' if q != 0.5 else UPSILON) for q in QUANTILES]
            x_c_cols = [(f'{X_C}_{q * 100:.1f}' if q != 0.5 else X_C) for q in QUANTILES]
            bdeiss_ids = {_[0] for _ in enumerate(model_ids) if MODELS[_[1]] in (BD, BDEI, BDSS, BDEISS)}
            for idx in bdeiss_ids:
                results[idx].loc[:, ups_cols] = np.zeros((n_forests, len(ups_cols)), dtype=float)
                results[idx].loc[:, x_c_cols] = np.ones((n_forests, len(ups_cols)), dtype=float)
        bdei_ids = {_[0] for _ in enumerate(model_ids) if 'EI' in MODELS[_[1]]}
        if bdei_ids and len(bdei_ids) < len(model_ids):
            f_cols = [(f'{F_E}_{q * 100:.1f}' if q != 0.5 else F_E) for q in QUANTILES]
            for idx in range(len(model_ids)):
                if not idx in bdei_ids:
                    results[idx].loc[:, f_cols] = np.zeros((n_forests, len(f_cols)), dtype=float)
        bdss_ids = {_[0] for _ in enumerate(model_ids) if 'SS' in MODELS[_[1]]}
        if bdss_ids and len(bdss_ids) < len(model_ids):
            f_cols = [(f'{F_S}_{q * 100:.1f}' if q != 0.5 else F_S) for q in QUANTILES]
            x_cols = [(f'{X_S}_{q * 100:.1f}' if q != 0.5 else X_S) for q in QUANTILES]
            for idx in range(len(model_ids)):
                if not idx in bdss_ids:
                    results[idx].loc[:, f_cols] = np.zeros((n_forests, len(f_cols)), dtype=float)
                    results[idx].loc[:, x_cols] = np.ones((n_forests, len(f_cols)), dtype=float)

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
    predict_parameters(forest_df, model_name=params.model_name, ci=params.ci, model_path=params.model_path_pattern)\
        .to_csv(params.log, header=True)


if '__main__' == __name__:
    main()