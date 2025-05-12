import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH
from bdeissct_dl.model_serializer import save_model_keras, save_scaler_joblib, save_scaler_numpy
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, QUANTILES, LA, PSI, RHO, UPSILON, X_C, KAPPA, F_E, F_S, \
    X_S, TARGET_COLUMNS_BDCT
from bdeissct_dl.pinball_loss import MultiQuantilePinballLoss

BATCH_SIZE = 4096

pd.set_option('display.max_columns', None)

EPOCHS = 10000


FEATURE_COLUMNS = [_ for _ in STATS if _ not in {'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 LA, PSI,
                                                 UPSILON, X_C, KAPPA,
                                                 F_E,
                                                 F_S, X_S,
                                                 SCALING_FACTOR}]


def build_model(n_x, n_y=4, optimizer=None, loss=None, metrics=None, quantiles=QUANTILES):
    """
    Build a FFNN of funnel shape (64-32-16-8 neurons), and a 4-neuron output layer (BD-CT unfixed parameters).
    We use a 50% dropout after each internal layer.
    This architecture follows teh PhyloDeep paper [Voznica et al. Nature 2022].

    :param n_x: input size (number of features)
    :param optimizer: by default Adam with learning rate 0f 0.001
    :param loss: loss function, by default MAPE
    :param metrics: evaluation metrics, by default ['accuracy', 'mape']
    :return: the model instance: tf.keras.models.Sequential
    """


    model = tf.keras.models.Sequential(name="FFNN")
    n_q = len(quantiles)
    n_out = n_y * n_q
    model.add(tf.keras.layers.InputLayer(shape=(n_x,), name='input_layer'))
    model.add(tf.keras.layers.Dense(n_out << 4, activation='elu', name=f'layer1_dense{n_out << 4}_elu'))
    model.add(tf.keras.layers.Dropout(0.5, name='dropout1_50'))
    model.add(tf.keras.layers.Dense(n_out << 3, activation='elu', name=f'layer2_dense{n_out << 3}_elu'))
    model.add(tf.keras.layers.Dropout(0.5, name='dropout2_50'))
    model.add(tf.keras.layers.Dense(n_out << 2, activation='elu', name=f'layer3_dense{n_out << 2}_elu'))
    # model.add(tf.keras.layers.Dropout(0.5, name='dropout3_50'))
    model.add(tf.keras.layers.Dense(n_out << 1, activation='elu', name=f'layer4_dense{n_out << 1}_elu'))
    model.add(tf.keras.layers.Dense(n_out, activation='linear', name=f'output_dense{n_out}_linear'))
    model.summary()

    if loss is None:
        loss = MultiQuantilePinballLoss(quantiles)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if metrics is None:
        metrics = ['accuracy']

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


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


def get_train_data(dfs=None, paths=None, target_columns=TARGET_COLUMNS_BDCT, scaler_x=None, scaler_y=None, is_training=True):

    if not dfs:
        dfs = [pd.read_csv(path) for path in paths]
    feature_columns = get_X_columns(dfs[0].columns)

    Xs, Ys, SFs = [], [], []
    for df in dfs:
        SFs.append(df.loc[:, SCALING_FACTOR].to_numpy(dtype=float, na_value=0))
        Xs.append(df.loc[:, feature_columns].to_numpy(dtype=float, na_value=0))
        Ys.append(df.loc[:, target_columns].to_numpy(dtype=float, na_value=0))

    X = np.concat(Xs, axis=0)
    Y = np.concat(Ys, axis=0)
    SF = np.concat(SFs, axis=0)

    if X.shape[0] > 1:
        n_examples = X.shape[0]
        permutation = np.random.choice(np.arange(n_examples), size=n_examples, replace=False)
        X = X[permutation, :]
        Y = Y[permutation, :]
        SF = SF[permutation]

    #Standardization of the input and output features with a standard scaler
    if scaler_x:
        X = scaler_x.fit_transform(X) if is_training else scaler_x.transform(X)
    if scaler_y:
        Y = scaler_y.fit_transform(Y) if is_training else scaler_y.transform(Y)


    return X, Y, SF


def fit_model(model, X_train_dev, Y_train_dev, val_fraction=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
    if val_fraction is None:
        val_fraction=calc_validation_fraction(X_train_dev.shape[0])

    #early stopping to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    #Training of the Network, with an independent validation set
    model.fit(X_train_dev, Y_train_dev, verbose=1, epochs=epochs, validation_split=val_fraction,
              batch_size=batch_size, callbacks=[early_stop])

    return model


def main():
    """
    Entry point for DL model training with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Train a BDCT model.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        help="path to the file where the encoded training data are stored")
    parser.add_argument('--model_name', required=True, type=str,
                        help="model name")
    parser.add_argument('--model_path', required=False, default=MODEL_PATH, type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path in the folder corresponding to the model name.")
    params = parser.parse_args()

    model_path = os.path.join(params.model_path, params.model_name)

    os.makedirs(model_path, exist_ok=True)
    scaler_x, scaler_y = StandardScaler(), StandardScaler() #MinMaxScaler()
    target_columns = MODEL2TARGET_COLUMNS[params.model_name]
    X, Y, SFs = get_train_data(paths=params.train_data, target_columns=target_columns,
                               scaler_x=scaler_x, scaler_y=scaler_y, is_training=True)

    model = build_model(n_x=X.shape[1], n_y=Y.shape[1])
    model = fit_model(model, X, Y)

    print(f'Saving the trained model to {model_path}...')

    save_model_keras(model, model_path)
    save_scaler_joblib(scaler_x, model_path, suffix='x')
    save_scaler_numpy(scaler_x, model_path, suffix='x')
    save_scaler_joblib(scaler_y, model_path, suffix='y')
    save_scaler_numpy(scaler_y, model_path, suffix='y')


if '__main__' == __name__:
    main()
