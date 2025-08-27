import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from bdeissct_dl import MODEL_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, LA, PSI, UPSILON, X_C, KAPPA, F_E, F_S, \
    X_S, TARGET_COLUMNS_BDCT, PI_E, PI_I, PI_S, PI_IC, PI_SC, PI_EC, PIS, UPS_X_C, F_S_X_S, BD, LA_AVG
from bdeissct_dl.dl_model import build_model
from bdeissct_dl.model_serializer import save_model_keras, load_scaler_numpy, \
    load_model_keras
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS

FEATURE_COLUMNS = [_ for _ in STATS if _ not in {'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 LA, PSI,
                                                 UPSILON, X_C, KAPPA,
                                                 F_E,
                                                 F_S, X_S,
                                                 PI_E, PI_I, PI_S,
                                                 PI_EC, PI_IC, PI_SC,
                                                 SCALING_FACTOR,
                                                 LA_AVG}]


def calc_validation_fraction(m):
    if m <= 1e4:
        return 0.2
    elif m <= 1e5:
        return 0.1
    return 0.01


def get_X_columns(columns):
    return FEATURE_COLUMNS


def get_test_data(dfs=None, paths=None, scaler_x=None):
    if not dfs:
        dfs = [pd.read_csv(path) for path in paths]
    feature_columns = get_X_columns(dfs[0].columns)

    Xs, SFs = [], []
    for df in dfs:
        SFs.append(df.loc[:, SCALING_FACTOR].to_numpy(dtype=float, na_value=0))
        Xs.append(df.loc[:, feature_columns].to_numpy(dtype=float, na_value=0))

    X = np.concat(Xs, axis=0)
    SF = np.concat(SFs, axis=0)

    # Standardization of the input features with a standard scaler
    if scaler_x:
        X = scaler_x.transform(X)

    return X, SF


def get_data_characteristics(paths, target_columns=TARGET_COLUMNS_BDCT):
    x_indices = []
    y_indices = []

    df = pd.read_csv(paths[0])
    feature_columns = set(get_X_columns(df.columns))
    target_columns = set(target_columns) if target_columns is not None else set()
    for i, col in enumerate(df.columns):
        if col in feature_columns:
            x_indices.append(i)
        if col in target_columns:
            y_indices.append(i)
    return x_indices, y_indices


def get_train_data(target_columns, columns_x, columns_y, file_pattern=None, filenames=None, scaler_x=None, \
                   batch_size=BATCH_SIZE, shuffle=False):

    if file_pattern is not None:
        filenames = glob.glob(filenames)

    Xs, Ys = [], []
    for path in filenames:
        try:
            df = pd.read_csv(path)
            Xs.append(df.iloc[:, columns_x].to_numpy(dtype=float, na_value=0))
            Ys.append(df.iloc[:, columns_y].to_numpy(dtype=float, na_value=0))
        except:
            print(f'Error reading file {path}. Skipping it.')
            continue

    X = np.concat(Xs, axis=0)
    Y = np.concat(Ys, axis=0)

    print('X has shape ', X.shape, 'Y has shape', Y.shape)

    if shuffle and X.shape[0] > 1:
        n_examples = X.shape[0]
        permutation = np.random.choice(np.arange(n_examples), size=n_examples, replace=False)
        X = X[permutation, :]
        Y = Y[permutation, :]

    # Standardization of the input and output features with a standard scaler
    if scaler_x:
        X = scaler_x.transform(X)

    train_labels = {
        LA: Y[:, 0],
        PSI: Y[:, 1],
    }
    col_i = 2
    if UPSILON in target_columns:
        train_labels[UPS_X_C] = Y[:, col_i: (col_i + 2)]
        col_i += 2
    if F_E in target_columns:
        train_labels[F_E] = Y[:, col_i]
        col_i += 1
    if F_S in target_columns:
        train_labels[F_S_X_S] = Y[:, col_i: (col_i + 2)]
        col_i += 2
    if PI_I in target_columns:
        train_labels[PIS] = Y[:, col_i:]

    dataset = tf.data.Dataset.from_tensor_slices((X, train_labels))

    dataset = (
        dataset
        # .shuffle(buffer_size=batch_size >> 1)  # Adjust buffer_size as appropriate
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def main():
    """
    Entry point for DL model training with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Train a BD(EI)(SS)(CT) model.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/BD/{i}/trees.csv.xz' for i in range(120)]
                        ,
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--val_data', type=str, nargs='+',
                        default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/BD/{i}/trees.csv.xz' for i in range(120, 128)]
                        ,
                        help="path to the files where the encoded validation data are stored")

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs to train the model')
    parser.add_argument('--base_model_name', type=str, default=None,
                        help="base model name to use for training, if not specified, the model will be trained from scratch")
    parser.add_argument('--model_name',
                        default=BD,
                        type=str, help="model name")
    parser.add_argument('--model_path', default=MODEL_PATH, type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path in the folder corresponding to the model name.")
    params = parser.parse_args()

    os.makedirs(params.model_path, exist_ok=True)

    target_columns = MODEL2TARGET_COLUMNS[params.model_name]
    # reshuffle params.train_data order
    if len(params.train_data) > 1:
        np.random.shuffle(params.train_data)
    if len(params.val_data) > 1:
        np.random.shuffle(params.val_data)


    x_indices, y_indices = get_data_characteristics(paths=params.train_data, target_columns=target_columns)

    scaler_x = load_scaler_numpy(params.model_path, suffix='x')

    if params.base_model_name is not None:
        model = load_model_keras(params.model_path, params.base_model_name)
        print(f'Loaded base model {params.base_model_name} with {len(x_indices)} input features and {len(y_indices)} output features.')
    else:
        model = build_model(target_columns, n_x=len(x_indices))
        print(f'Building a model from scratch with {len(x_indices)} input features and {len(y_indices)} output features.')
    print(model.summary())

    ds_train = get_train_data(target_columns, x_indices, y_indices, file_pattern=None, filenames=params.train_data, \
                              scaler_x=scaler_x, batch_size=BATCH_SIZE * 8, shuffle=True)
    ds_val = get_train_data(target_columns, x_indices, y_indices, file_pattern=None, filenames=params.val_data, \
                            scaler_x=scaler_x, batch_size=BATCH_SIZE * 8, shuffle=True)

    #early stopping to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    #Training of the Network, with an independent validation set
    model.fit(ds_train, verbose=1, epochs=params.epochs, validation_data=ds_val,
              callbacks=[early_stop])

    print(f'Saving the trained model {params.model_name} to {params.model_path}...')

    save_model_keras(model, path=params.model_path, model_name=params.model_name)



if '__main__' == __name__:
    main()
