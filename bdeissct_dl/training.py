import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from bdeissct_dl import MODEL_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, UPSILON, X_C, KAPPA, INCUBATION_PERIOD, F_S, \
    X_S, TARGET_COLUMNS_BDCT, REPRODUCTIVE_NUMBER, INFECTION_DURATION
from bdeissct_dl.dl_model import build_model
from bdeissct_dl.model_serializer import save_model_keras, load_scaler_numpy, \
    load_model_keras
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS

FEATURE_COLUMNS = [_ for _ in STATS if _ not in {'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 REPRODUCTIVE_NUMBER, INFECTION_DURATION,
                                                 UPSILON, X_C, KAPPA,
                                                 INCUBATION_PERIOD,
                                                 F_S, X_S,
                                                 SCALING_FACTOR}]


def calc_validation_fraction(m):
    if m <= 1e4:
        return 0.2
    elif m <= 1e5:
        return 0.1
    return 0.01


def get_test_data(dfs=None, paths=None, scaler_x=None):
    if not dfs:
        dfs = [pd.read_csv(path) for path in paths]
    feature_columns = FEATURE_COLUMNS

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


def get_data_characteristics(paths, target_columns=TARGET_COLUMNS_BDCT, feature_columns=FEATURE_COLUMNS):
    col2index_y = {}
    col2index_x = {}

    df = pd.read_csv(paths[0])
    feature_column_set = set(feature_columns)
    target_columns = target_columns if target_columns is not None else []
    target_column_set = set(target_columns)
    for i, col in enumerate(df.columns):
        if col in feature_column_set:
            col2index_x[col] = i
        if col in target_column_set:
            col2index_y[col] = i
    return [col2index_x[_] for _ in feature_columns], col2index_y


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

    train_labels = {}
    col_i = 0
    if REPRODUCTIVE_NUMBER in target_columns:
        train_labels[REPRODUCTIVE_NUMBER] = Y[:, col_i]
        col_i += 1
    if INFECTION_DURATION in target_columns:
        train_labels[INFECTION_DURATION] = Y[:, col_i]
        col_i += 1
    if UPSILON in target_columns:
        train_labels[UPSILON] = Y[:, col_i]
        col_i += 1
    if X_C in target_columns:
        train_labels[X_C] = Y[:, col_i]
        col_i += 1
    if INCUBATION_PERIOD in target_columns:
        train_labels[INCUBATION_PERIOD] = Y[:, col_i]
        col_i += 1
    if F_S in target_columns:
        train_labels[F_S] = Y[:, col_i]
        col_i += 1
    if X_S in target_columns:
        train_labels[X_S] = Y[:, col_i]
        col_i += 1

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
                        # default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train/2000_5000/BDEI/{i}/trees.csv.xz' for i in range(100)] \
                        #         + [f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/2000_5000/BD/{i}/trees.csv.xz' for i in range(10)]
                        # ,
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--val_data', type=str, nargs='+',
                        # default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train/2000_5000/BDEI/{i}/trees.csv.xz' for i in range(100, 120)] \
                        #     + [f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train/2000_5000/BD/{i}/trees.csv.xz' for i in range(10, 12)]
                        # ,
                        help="path to the files where the encoded validation data are stored")

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs to train the model')
    parser.add_argument('--base_model_name', type=str, default=None,
                        help="base model name to use for training, if not specified, the model will be trained from scratch")
    parser.add_argument('--model_name',
                        # default=BDEI,
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


    x_indices, y_col2index = get_data_characteristics(paths=params.train_data, target_columns=target_columns)

    scaler_x = load_scaler_numpy(params.model_path, suffix='x')


    for col, y_idx in y_col2index.items():
        print(f'Training to predict {col} with {params.model_name}...')

        if params.base_model_name is not None:
            model = load_model_keras(params.model_path, f'{params.base_model_name}.{col}')
            print(f'Loaded base model {params.base_model_name} with {len(x_indices)} input features and {col} as output.')
        else:
            model = build_model([col], n_x=len(x_indices))
            print(f'Building a model from scratch with {len(x_indices)} input features and {col} as output.')
        print(model.summary())

        ds_train = get_train_data([col], x_indices, [y_idx], file_pattern=None, filenames=params.train_data, \
                                  scaler_x=scaler_x, batch_size=BATCH_SIZE, shuffle=True)
        ds_val = get_train_data([col], x_indices, [y_idx], file_pattern=None, filenames=params.val_data, \
                                scaler_x=scaler_x, batch_size=BATCH_SIZE, shuffle=True)

        #early stopping to avoid overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

        #Training of the Network, with an independent validation set
        model.fit(ds_train, verbose=1, epochs=params.epochs, validation_data=ds_val, callbacks=[early_stop])

        print(f'Saving the trained model {params.model_name}.{col} to {params.model_path}...')
        save_model_keras(model, path=params.model_path, model_name=f'{params.model_name}.{col}')



if '__main__' == __name__:
    main()
