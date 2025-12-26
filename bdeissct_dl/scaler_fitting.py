import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.bdeissct_model import TARGET_COLUMNS_BDEISSCT
from bdeissct_dl.model_serializer import save_scaler_numpy
from bdeissct_dl.training import get_data_characteristics


def fit_scalers(paths, x_indices, scaler_x=None):
   for path in paths:
        df = pd.read_csv(path)
        if scaler_x:
            X = df.iloc[:, x_indices].to_numpy(dtype=float, na_value=0)
            scaler_x.fit(X)


def main():
    """
    Entry point for BDEISSCT data scaling with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Fit a BD(EI)(SS)(CT) data scaler.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        # default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/{model}/{i}/trees.csv.xz' for i in range(120) for model in [BD, BDCT, BDEI, BDEICT, BDSS, BDSSCT, BDEISS, BDEISSCT]],
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--model_path', default=MODEL_PATH, type=str,
                        help="path to the folder where the scaler should be stored.")
    params = parser.parse_args()

    os.makedirs(params.model_path, exist_ok=True)

    scaler_x = StandardScaler()
    x_indices, _ = \
        get_data_characteristics(paths=params.train_data, target_columns=TARGET_COLUMNS_BDEISSCT)
    fit_scalers(paths=params.train_data, x_indices=x_indices, scaler_x=scaler_x)

    if scaler_x is not None:
        save_scaler_numpy(scaler_x, params.model_path, suffix='x')


if '__main__' == __name__:
    main()
