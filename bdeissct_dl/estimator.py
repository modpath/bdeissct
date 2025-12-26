import pandas as pd

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, BD, MODELS
from bdeissct_dl.model_serializer import load_model_keras, load_scaler_numpy
from bdeissct_dl.training import get_test_data
from bdeissct_dl.tree_encoder import forest2sumstat_df, scale_back
from bdeissct_dl.tree_manager import read_forest


def predict_parameters(forest_sumstats, model_name=BD, model_path=MODEL_PATH):
    scaler_x = load_scaler_numpy(model_path, suffix='x')
    X, SF = get_test_data(dfs=[forest_sumstats], scaler_x=scaler_x)

    target_columns = MODEL2TARGET_COLUMNS[model_name]

    result = None
    for col in target_columns:
        model = load_model_keras(model_path, f'{model_name}.{col}')
        Y_pred = model.predict(X)

        if len(Y_pred[col].shape) == 2 and Y_pred[col].shape[1] == 1:
            Y_pred[col] = Y_pred[col].squeeze(axis=1)

        scale_back(Y_pred, SF)
        res_df = pd.DataFrame.from_dict(Y_pred, orient='columns')
        result = result.join(res_df, how='outer') if result is not None else res_df

    return result


def main():
    """
    Entry point for tree parameter estimation with a BDCT model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimate BD(EI)(SS)(CT) model parameters.")
    parser.add_argument('--model_name', choices=MODELS, default=BD, type=str,
                        help=f'BDEISSCT model flavour')
    parser.add_argument('--model_path', default=MODEL_PATH,
                        help='By default our pretrained BD(EI)(SS)(CT) models are used, '
                             'but it is possible to specify a path to a custom folder here, '
                             'containing files "<model_name>.keras" (with the model), '
                             'and scaler-related files to rescale the input data X: '
                             '"data_scalerx_mean.npy", "data_scalerx_scale.npy", "data_scalerx_var.npy" '
                             '(unpickled numpy-saved arrays), '
                             'and "data_scalerx_n_samples_seen.txt" '
                             'a text file containing the number of examples in the training set).'
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
    predict_parameters(forest_df, model_name=params.model_name, model_path=params.model_path)\
        .to_csv(params.log, header=True)


if '__main__' == __name__:
    main()