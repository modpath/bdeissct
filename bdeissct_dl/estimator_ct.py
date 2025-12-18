import numpy as np
import pandas as pd

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.bdeissct_model import CT_EPI_COLUMNS, CT_RATE_COLUMNS, PSI, REPRODUCTIVE_NUMBER, LA, X_C
from bdeissct_dl.model_serializer import load_model_keras, load_scaler_numpy


def predict_parameters(df, model_path=MODEL_PATH):
    feature_columns = CT_EPI_COLUMNS
    X = df.loc[:, feature_columns].to_numpy(dtype=float, na_value=0)

    # Standardization of the input features with a
    # standard scaler
    scaler_x = load_scaler_numpy(model_path, suffix='ct.x')
    if scaler_x:
        X = scaler_x.transform(X)

    target_columns = CT_RATE_COLUMNS

    result = None
    for col in target_columns:
        model = load_model_keras(model_path, f'CT.{col}')
        Y_pred = model.predict(X)

        if len(Y_pred[col].shape) == 2 and Y_pred[col].shape[1] == 1:
            Y_pred[col] = Y_pred[col].squeeze(axis=1)

        res_df = pd.DataFrame.from_dict(Y_pred, orient='columns')
        result = result.join(res_df, how='outer') if result is not None else res_df
    if PSI in result.columns:
        d_I_effective = df[REPRODUCTIVE_NUMBER] / df[LA]
        psi_effective = 1 / d_I_effective
        # psi_effective = alpha  psi + (1 - alpha) X_C psi
        # then psi = psi_effective / (alpha + (1 - alpha) X_C)
        result[PSI] = np.maximum(np.minimum(result[PSI], psi_effective), psi_effective / df[X_C])

    return result


def main():
    """
    Entry point for tree parameter estimation with a BDCT model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Estimate CT rates from other model parameters.")
    parser.add_argument('--model_path', default=MODEL_PATH,
                        help='By default our pretrained CT model is used, '
                             'but it is possible to specify a path to a custom folder here, '
                             'containing files "CT.keras" (with the model), '
                             'and scaler-related files to rescale the input data X, and the output Y: '
                             'for X: "data_scalerct.x_mean.npy", "data_scalerct.x_scale.npy", "data_scalerct.x_var.npy" '
                             '(unpickled numpy-saved arrays), '
                             'and "data_scalerct.x_n_samples_seen.txt" '
                             'a text file containing the number of examples in the training set). '
                             'For Y the file names are the same, just x replaced by y, e.g., "data_scalerct.y_mean.npy". '
                             )
    parser.add_argument('--log', default=None, type=str, help="output log file")
    parser.add_argument('--sumstats', default=None, type=str, help="input file(s) with epi parameters")
    params = parser.parse_args()

    df = pd.read_csv(params.sumstats)
    predict_parameters(df, model_path=params.model_path).to_csv(params.log, header=True)


if '__main__' == __name__:
    main()