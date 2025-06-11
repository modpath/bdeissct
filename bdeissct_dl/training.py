import glob
import os
import lzma

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from wheel.metadata import yield_lines

from bdeissct_dl import MODEL_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, LA, PSI, UPSILON, X_C, KAPPA, F_E, F_S, \
    X_S, TARGET_COLUMNS_BDCT, PI_E, PI_I, PI_S, PI_IC, PI_SC, PI_EC, BD, BDSS
from bdeissct_dl.model_serializer import save_model_keras, save_scaler_joblib, save_scaler_numpy, load_scaler_numpy, \
    load_model_keras
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS
from bdeissct_dl.dl_model import build_model

FEATURE_COLUMNS = [_ for _ in STATS if _ not in {'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 LA, PSI,
                                                 UPSILON, X_C, KAPPA,
                                                 F_E,
                                                 F_S, X_S,
                                                 PI_E, PI_I, PI_S,
                                                 PI_EC, PI_IC, PI_SC,
                                                 SCALING_FACTOR}]


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


def get_data_characteristics(paths, target_columns=TARGET_COLUMNS_BDCT, scaler_x=None, scaler_y=None):
    x_indices = []
    y_indices = []
    n_col = 0

    # First pass: calculate mean and var
    for path in paths:
        df = pd.read_csv(path)
        if not x_indices:
            feature_columns = set(get_X_columns(df.columns))
            target_columns = set(target_columns) if target_columns is not None else set()
            n_col = len(df.columns)
            for i, col in enumerate(df.columns):
                # fix the bug from prev. version of encoding
                if col == 'pi_E.1':
                    col = 'pi_EC'
                if col in feature_columns:
                    x_indices.append(i)
                if col in target_columns:
                    y_indices.append(i)
            if scaler_x is None and scaler_y is None:
                # no need to fit scalers and hence no need to see the rest of the data
                break
        if scaler_x:
            X = df.iloc[:, x_indices].to_numpy(dtype=float, na_value=0)
            scaler_x.partial_fit(X)
        if scaler_y:
            Y = df.iloc[:, y_indices].to_numpy(dtype=float, na_value=0)
            scaler_y.partial_fit(Y)
    return x_indices, y_indices, n_col

def get_train_data(n_input, columns_x, columns_y, file_pattern=None, filenames=None, scaler_x=None, \
                   batch_size=BATCH_SIZE, shuffle=False):

    def parse_line(line):
        """
        parse a single line
        :param line:
        :return:
        """
        # decode into a tensor with default values (if something is missing in the given dataframe line) set to 0
        fields = tf.io.decode_csv(line, [0.0] * n_input, field_delim=",", use_quote_delim=False)
        X = tf.stack([fields[i] for i in columns_x], axis=-1)
        Y = tf.stack([fields[i] for i in columns_y], axis=-1)

        train_labels = {
            "la": Y[0],
            "psi": Y[1],
            "ups": Y[2],
            "X_C": Y[3],
            "f_E": Y[4],
            "f_S": Y[5],
            "X_S": Y[6],
            "pi": Y[7:]
        }


        return X, train_labels


    if file_pattern is not None:
        filenames = glob.glob(filenames)

    def read_xz_lines(filenames):

        for filename in filenames:
            # Opens .xz file for reading text (line by line)
            with lzma.open(filename, "rt") as f:
                # skip the header
                next(f)
                for line in f:
                    line = line.strip()
                    if line:
                        # print([_ for (i, _) in enumerate(line.split(',')) if i in columns_y])
                        yield line

    dataset = tf.data.Dataset.from_generator(
        lambda: read_xz_lines(filenames),
        output_types=tf.string,  # each line is a string
        output_shapes=()
    )

    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.AUTOTUNE)

    def scale(x, y):
        if scaler_x:
            mean_x, scale_x = tf.constant(scaler_x.mean_, dtype=tf.float32), tf.constant(scaler_x.scale_, dtype=tf.float32)
            x = (x - mean_x) / scale_x
        return x, y

    dataset = dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE)

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
                        default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/BDSS/{i}/trees.csv.xz' for i in range(20)],
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--val_data', type=str, nargs='+',
                        default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/BDSS/{i}/trees.csv.xz' for i in range(124, 128)],
                        help="path to the files where the encoded validation data are stored")

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs to train the model')
    parser.add_argument('--base_model_name', type=str, default=None,
                        help="base model name to use for training, if not specified, the model will be trained from scratch")
    parser.add_argument('--model_name', type=str, help="model name")
    parser.add_argument('--model_path', default=None, type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path in the folder corresponding to the model name.")
    params = parser.parse_args()

    if params.model_path is None:
        model_path = MODEL_PATH.format(params.model_name)
        base_model_path = MODEL_PATH.format(params.base_model_name) if params.base_model_name else None
    else:
        model_path = os.path.join(params.model_path, params.model_name)
        base_model_path = os.path.join(params.model_path, params.base_model_name) if params.base_model_name else None

    os.makedirs(model_path, exist_ok=True)

    target_columns = MODEL2TARGET_COLUMNS[params.model_name]
    # reshuffle params.train_data order
    if len(params.train_data) > 1:
        np.random.shuffle(params.train_data)
    if len(params.val_data) > 1:
        np.random.shuffle(params.val_data)

    if params.base_model_name is not None:
        scaler_x = load_scaler_numpy(base_model_path, suffix='x')
        scaler_y = load_scaler_numpy(base_model_path, suffix='y')
        x_indices, y_indices, n_columns = \
            get_data_characteristics(paths=params.train_data, target_columns=target_columns, \
                                     scaler_x=None, scaler_y=None)
        model = load_model_keras(base_model_path)
        print(f'Loaded base model from {base_model_path} with {len(x_indices)} input features and {len(y_indices)} output features.')
        print(model.summary())
    else:
        scaler_x, scaler_y = StandardScaler(), None
        x_indices, y_indices, n_columns = \
            get_data_characteristics(paths=params.train_data, target_columns=target_columns, \
                                     scaler_x=scaler_x, scaler_y=scaler_y)
        model = build_model(n_x=len(x_indices), n_y=len(y_indices))



    ds_train = get_train_data(n_columns, x_indices, y_indices, file_pattern=None, filenames=params.train_data, \
                              scaler_x=scaler_x, batch_size=BATCH_SIZE * 8, shuffle=False)
    ds_val = get_train_data(n_columns, x_indices, y_indices, file_pattern=None, filenames=params.val_data, \
                            scaler_x=scaler_x, batch_size=BATCH_SIZE * 8, shuffle=False)



    #early stopping to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    #Training of the Network, with an independent validation set
    model.fit(ds_train, verbose=1, epochs=params.epochs, validation_data=ds_val,
              callbacks=[early_stop])

    print(f'Saving the trained model to {model_path}...')

    save_model_keras(model, model_path)

    if scaler_x is not None:
        save_scaler_joblib(scaler_x, model_path, suffix='x')
        save_scaler_numpy(scaler_x, model_path, suffix='x')
    if scaler_y is not None:
        save_scaler_joblib(scaler_y, model_path, suffix='y')
        save_scaler_numpy(scaler_y, model_path, suffix='y')


if '__main__' == __name__:
    main()
