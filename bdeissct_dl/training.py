import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, UPSILON, X_C, KAPPA, INCUBATION_FRACTION, F_S, \
    X_S, TARGET_COLUMNS_BDCT, REPRODUCTIVE_NUMBER, INFECTION_DURATION
from bdeissct_dl.dl_model import build_model
from bdeissct_dl.model_serializer import save_model_keras, load_scaler_numpy, \
    load_model_keras, save_scaler_numpy
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS

FEATURE_COLUMNS = [_ for _ in STATS if _ not in {#'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 REPRODUCTIVE_NUMBER, INFECTION_DURATION,
                                                 UPSILON, X_C, KAPPA,
                                                 INCUBATION_FRACTION,
                                                 F_S, X_S,
                                                 SCALING_FACTOR}]


def get_train_data(target_columns, columns_x, columns_y, file_pattern=None, filenames=None,
                   scaler_x=None, scaler_y=None,
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
    if scaler_y:
        Y = scaler_y.transform(Y)

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
    if INCUBATION_FRACTION in target_columns:
        train_labels[INCUBATION_FRACTION] = Y[:, col_i]
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


def train_column_models(params, scaler_x, scaler_y, x_indices, y_col2index):
    for col, y_idx in y_col2index.items():
        try:
            if load_model_keras(path=params.model_path, model_name=f'{params.model_name}.{col}'):
                print(
                    f'Model {params.model_name}.{col} already exists at {params.model_path}. Skipping training for this target.')
                continue
        except:
            pass

        print(f'Training to predict {col} with {params.model_name}...')

        if params.base_model_name is not None:
            model = load_model_keras(params.model_path, f'{params.base_model_name}.{col}')
            print(
                f'Loaded base model {params.base_model_name} with {len(x_indices)} input features and {col} as output.')
        else:
            model = build_model([col], n_x=len(x_indices))
            print(f'Building a model from scratch with {len(x_indices)} input features and {col} as output.')
        print(model.summary())

        ds_train = get_train_data([col], x_indices, [y_idx], filenames=params.train_data,
                                  scaler_x=scaler_x, scaler_y=scaler_y, batch_size=BATCH_SIZE, shuffle=True)
        ds_val = get_train_data([col], x_indices, [y_idx], filenames=params.val_data,
                                scaler_x=scaler_x, scaler_y=scaler_y, batch_size=BATCH_SIZE, shuffle=True)

        # early stopping to avoid overfitting

        # early stopping to avoid overfitting
        early_stop = get_esrly_stopping()
        reduce_lr = get_learning_rate_scaler()

        # Training of the Network, with an independent validation set
        history = model.fit(ds_train, verbose=1, epochs=params.epochs, validation_data=ds_val,
                            callbacks=[early_stop, reduce_lr])

        plot_losses(os.path.join(params.model_path, f'{params.model_name}.{col}.pdf'), history, model, ds_train, ds_val,
                    best_epoch=early_stop.best_epoch)

        print(f'Saving the trained model {params.model_name}.{col} to {params.model_path}...')
        save_model_keras(model, path=params.model_path, model_name=f'{params.model_name}.{col}')


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


def train_model(params, scaler_x, scaler_y, x_indices, y_indices, target_columns):

    print(f'Training a {params.model_name} estimator...')

    if params.base_model_name is not None:
        model = load_model_keras(params.model_path, f'{params.base_model_name}')
        print(
            f'Loaded base model {params.base_model_name}.')
    else:
        model = build_model(target_columns, n_x=len(x_indices))
        print(f'Building a model from scratch with {len(x_indices)} input features and {len(target_columns)} as output.')
    print(model.summary())

    ds_train = get_train_data(target_columns, x_indices, y_indices, file_pattern=None, filenames=params.train_data, \
                              scaler_x=scaler_x, scaler_y=scaler_y, batch_size=BATCH_SIZE, shuffle=True)
    ds_val = get_train_data(target_columns, x_indices, y_indices, file_pattern=None, filenames=params.val_data, \
                            scaler_x=scaler_x, scaler_y=scaler_y, batch_size=BATCH_SIZE, shuffle=True)

    # early stopping to avoid overfitting
    early_stop = get_esrly_stopping()
    reduce_lr = get_learning_rate_scaler()

    # Training of the Network, with an independent validation set
    history = model.fit(ds_train, verbose=1, epochs=params.epochs, validation_data=ds_val,
                        callbacks=[early_stop, reduce_lr])

    plot_losses(os.path.join(params.model_path, f'{params.model_name}.pdf'), history, model, ds_train, ds_val,
                best_epoch=early_stop.best_epoch)

    print(f'Saving the trained model {params.model_name} to {params.model_path}...')
    save_model_keras(model, path=params.model_path, model_name=f'{params.model_name}')


def get_esrly_stopping() -> tf.keras.callbacks.EarlyStopping:
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
    parser.add_argument('--base_model_name', type=str, default=None,
                        help="base model name to use for training, if not specified, the model will be trained from scratch")
    parser.add_argument('--model_name', type=str, help="model name")
    parser.add_argument('--model_path', default=MODEL_PATH, type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path in the file <model name>.keras.")
    parser.add_argument('--per_target', action='store_true',
                        help="Train separate models for each target parameter "
                             "instead of a single model for all target parameters.")
    params = parser.parse_args()

    os.makedirs(params.model_path, exist_ok=True)

    target_columns = MODEL2TARGET_COLUMNS[params.model_name]
    # reshuffle params.train_data order
    if len(params.train_data) > 1:
        np.random.shuffle(params.train_data)
    if len(params.val_data) > 1:
        np.random.shuffle(params.val_data)


    x_indices, y_col2index = get_data_characteristics(paths=params.train_data, target_columns=target_columns)
    y_indices = [y_col2index[_] for _ in target_columns]

    scaler_x = load_scaler_numpy(params.model_path, suffix=f'{params.model_name}.x')
    scaler_y = load_scaler_numpy(params.model_path, suffix=f'{params.model_name}.y')
    if scaler_x is None or scaler_y is None:
        from bdeissct_dl.scaler_fitting import fit_scalers
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        fit_scalers(paths=params.train_data, x_indices=x_indices, scaler_x=scaler_x,
                    y_indices=y_indices, scaler_y=scaler_y)

        if scaler_x is not None:
            save_scaler_numpy(scaler_x, params.model_path, suffix=f'{params.model_name}.x')
        if scaler_y is not None:
            save_scaler_numpy(scaler_y, params.model_path, suffix=f'{params.model_name}.y')

    if params.per_target:
        train_column_models(params, scaler_x, scaler_y=scaler_y, x_indices=x_indices, y_col2index=y_col2index)
    else:
        train_model(params, scaler_x=scaler_x, scaler_y=scaler_y, x_indices=x_indices,
                    y_indices=y_indices, target_columns=target_columns)

if '__main__' == __name__:
    main()
