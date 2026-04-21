import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, UPSILON, X_C, KAPPA, INCUBATION_FRACTION, F_S, \
    X_S, TARGET_COLUMNS_BDCT, REPRODUCTIVE_NUMBER, INFECTION_DURATION
from bdeissct_dl.dl_model import get_model_layers, get_outputs_pinball, pinball_loss, QUANTILES
from bdeissct_dl.model_serializer import save_model_keras, load_scaler_numpy, save_scaler_numpy, RANDOM_SEED
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS

FEATURE_COLUMNS = [_ for _ in STATS if _ not in {#'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 REPRODUCTIVE_NUMBER, INFECTION_DURATION,
                                                 UPSILON, X_C, KAPPA,
                                                 INCUBATION_FRACTION,
                                                 F_S, X_S,
                                                 SCALING_FACTOR}]


LEARNING_RATE = 0.01
EPOCHS = 1000
BATCH_SIZE = 8192

def fit_scalers(paths, x_indices, scaler_x=None, y_indices=None, scaler_y=None):
   for path in paths:
        df = pd.read_csv(path)
        if scaler_x:
            X = df.iloc[:, x_indices].to_numpy(dtype=float, na_value=0)
            scaler_x.partial_fit(X)
        if scaler_y:
            Y = df.iloc[:, y_indices].to_numpy(dtype=float, na_value=0)
            scaler_y.partial_fit(Y)

def get_scalers(model_name, train_data, model_path, x_indices, y_indices=None, scale_y=True):
    scaler_x = load_scaler_numpy(model_path, suffix=f'{model_name}.x')
    scaler_y = None if not scale_y else load_scaler_numpy(model_path, suffix=f'{model_name}.y')
    if scaler_x is None or (scaler_y and scaler_y is None):
        scaler_x = StandardScaler()
        scaler_y = None if not scale_y else StandardScaler()
        fit_scalers(paths=train_data, x_indices=x_indices, scaler_x=scaler_x,
                    y_indices=y_indices, scaler_y=scaler_y)

        if scaler_x is not None:
            save_scaler_numpy(scaler_x, model_path, suffix=f'{model_name}.x')
        if scaler_y is not None:
            save_scaler_numpy(scaler_y, model_path, suffix=f'{model_name}.y')
    return scaler_x, scaler_y


def get_train_data(target_columns, columns_x, columns_y, file_pattern=None, filenames=None,
                   scaler_x=None, scaler_y=None, shuffle=False):

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
    if scaler_y:
        Y = scaler_y.transform(Y)

    train_labels = {col: Y[:, col_i] for (col_i, col) in enumerate(target_columns)}
    return X, train_labels


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


def plot_losses(pdf, history, model, train_ds, val_ds, best_epoch: int):
    train_loss = model.evaluate(train_ds, verbose=0)
    val_loss = model.evaluate(val_ds, verbose=0)

    loss_names = model.metrics_names  # ['loss', 'R_loss', 'X_C_loss', ...]

    # Format with labels
    if len(loss_names) > 1:
        train_str = ', '.join([f'{name}: {loss:.3f}' for name, loss in zip(loss_names, train_loss)])
        val_str = ', '.join([f'{name}: {loss:.3f}' for name, loss in zip(loss_names, val_loss)])
    else:
        train_str = f'{train_loss:.3f}'
        val_str = f'{val_loss:.3f}'

    print(f"Training Loss: {train_str}")
    print(f"Validation Loss: {val_str}")


    # Plot training & validation loss
    epochs_to_skip = 5  # so the initial loss is not shown, as it is usually very high and makes the plot less readable

    plt.figure(figsize=(10, 6))
    epochs_range = range(epochs_to_skip + 1, len(history.history['loss']) + 1)
    plt.plot(epochs_range, history.history['loss'][epochs_to_skip:], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'][epochs_to_skip:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Vertical line
    plt.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7)
    plt.scatter(best_epoch + 1, history.history['val_loss'][best_epoch],
                color='red', s=100, zorder=5, marker='*')

    plt.legend()

    plt.title(f'Train {train_str} vs Val {val_str}')
    plt.savefig(pdf, dpi=100)


def get_early_stopping() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Important! Restore best weights
        min_delta=1e-3,  # Minimum change to qualify as improvement
        verbose=1  # See when it triggers
    )


def get_learning_rate_scaler() -> tf.keras.callbacks.ReduceLROnPlateau:
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )


def main():
    """
    Entry point for DL model training with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Train a BD(EI)(SS)(CT) model.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--val_data', type=str, nargs='+',
                        help="path to the files where the encoded validation data are stored")

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs to train the model')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='if a non-negative number is given, '
                                                             'it will be set as a random seed.')
    parser.add_argument('--model_name', type=str, help="model name")
    parser.add_argument('--model_path', type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path in the file <model name>.keras if no random seed is given."
                             "If the random seed is given, it will be saved as <model name>.<seed>.keras instead.")
    params = parser.parse_args()

    train_main(**vars(params))


def train_main(model_name, train_data, val_data, model_path, epochs=EPOCHS, seed=RANDOM_SEED):
    os.makedirs(model_path, exist_ok=True)

    target_columns = MODEL2TARGET_COLUMNS[model_name]
    # reshuffle train_data order
    if len(train_data) > 1:
        np.random.shuffle(train_data)
    if len(val_data) > 1:
        np.random.shuffle(val_data)

    x_indices, y_col2index = get_data_characteristics(paths=train_data, target_columns=target_columns)
    y_indices = [y_col2index[_] for _ in target_columns]

    scaler_x, scaler_y = get_scalers(model_name=model_name, train_data=train_data,
                                     x_indices=x_indices, y_indices=y_indices, model_path=model_path,
                                     scale_y=True)

    print(f'Training a {model_name} estimator...')
    if seed and seed > 0:
        print(f'Fixed the random seed to {seed}.')
        np.random.seed(239)
        tf.random.set_seed(239)

    inputs, x = get_model_layers(n_x=len(x_indices))
    outputs = get_outputs_pinball(target_columns, x, quantiles=QUANTILES)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss={col: pinball_loss for col in target_columns})

    print(f'Building a model from scratch with {len(x_indices)} input features and {len(target_columns)} as output.')
    print(model.summary())

    ds_train_X, ds_train_Y = get_train_data(target_columns, x_indices, y_indices, filenames=train_data,
                                            scaler_x=scaler_x, scaler_y=scaler_y, shuffle=True)
    ds_val_X, ds_val_Y = get_train_data(target_columns, x_indices, y_indices, filenames=val_data,
                                        scaler_x=scaler_x, scaler_y=scaler_y, shuffle=True)

    ds_train = tf.data.Dataset.from_tensor_slices((ds_train_X, ds_train_Y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((ds_val_X, ds_val_Y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # early stopping to avoid overfitting
    early_stop = get_early_stopping()
    reduce_lr = get_learning_rate_scaler()

    # Training of the Network, with an independent validation set
    history = model.fit(ds_train, verbose=1, epochs=epochs, validation_data=ds_val,
                        callbacks=[early_stop, reduce_lr])

    model_name = f'{model_name}.{seed}' if seed and seed > 0 else f'{model_name}'

    plot_losses(os.path.join(model_path, f'{model_name}.pdf'), history, model, ds_train, ds_val,
                best_epoch=early_stop.best_epoch)

    print(f'Saving the trained model {model_name} to {model_path}...')
    save_model_keras(model, path=model_path, model_name=f'{model_name}')


if '__main__' == __name__:
    main()
