import os
import lzma

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_FINDER_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import MODELS
from bdeissct_dl.model_serializer import save_model_keras, save_scaler_joblib, save_scaler_numpy, load_scaler_numpy
from bdeissct_dl.training import get_X_columns, calc_validation_fraction, get_data_characteristics


def build_model(n_x, n_y, optimizer=None, loss=None, metrics=None):
    """
    Build a FFNN of funnel shape (64-32-16-max(n_y, 8) neurons), and a n_y-neuron output layer (model probabilities).
    We use a 50% dropout.
    This architecture follows teh PhyloDeep paper [Voznica et al. Nature 2022].

    :param n_x: input size (number of features)
    :param optimizer: by default Adam with learning rate of 0.001
    :param loss: loss function, by default categorical crossentropy
    :param metrics: evaluation metrics, by default ['accuracy']
    :return: the model instance: tf.keras.models.Sequential
    """


    model = tf.keras.models.Sequential(name="FFNN_MF")
    model.add(tf.keras.layers.InputLayer(shape=(n_x,), name='input_layer'))
    model.add(tf.keras.layers.Dense(n_y << 4, activation='elu', name=f'layer1_dense{n_y << 4}_elu'))
    model.add(tf.keras.layers.Dropout(0.5, name='dropout1_50'))
    model.add(tf.keras.layers.Dense(n_y << 3, activation='elu', name=f'layer2_dense{n_y << 3}_elu'))
    model.add(tf.keras.layers.Dropout(0.5, name='dropout2_50'))
    model.add(tf.keras.layers.Dense(n_y << 2, activation='elu', name=f'layer3_dense{n_y << 2}_elu'))
    # model.add(tf.keras.layers.Dropout(0.5, name='dropout3_50'))
    model.add(tf.keras.layers.Dense(n_y << 1, activation='elu', name=f'layer4_dense{n_y << 1}_elu'))
    model.add(tf.keras.layers.Dense(n_y, activation='softmax', name=f'output_dense{n_y}_softmax'))
    model.summary()

    if loss is None:
        loss = tf.keras.losses.CategoricalCrossentropy()
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if metrics is None:
        metrics = ['accuracy']

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def get_test_data(df=None, path=None):
    if df is None:
        df = pd.read_csv(path)
    feature_columns = get_X_columns(df.columns)
    X = df.loc[:, feature_columns].to_numpy(dtype=float, na_value=0)
    # Standardization of the input features with a standard scaler
    X = load_scaler_numpy(MODEL_FINDER_PATH, suffix='x').transform(X)
    return X


def get_train_data(n_input, columns_x, labels, file_pattern=None, filenames=None, scaler_x=None, \
                   batch_size=BATCH_SIZE, shuffle=False):
    n_y = len(MODELS)
    y_line = ','.join('0' for _ in range(n_y))

    def parse_line(line):
        """
        parse a single line
        :param line:
        :return:
        """
        # decode into a tensor with default values (if something is missing in the given dataframe line) set to 0
        fields = tf.io.decode_csv(line, [0.0] * (n_input + n_y), field_delim=",", use_quote_delim=False)
        X = tf.stack([fields[i] for i in columns_x], axis=-1)
        Y = tf.stack(fields[n_input:], axis=-1)
        return X, Y


    if file_pattern is not None:
        filenames = glob.glob(filenames)

    def read_xz_lines(filenames, labels, y_line):
        for filename, label in zip(filenames, labels):
            # Opens .xz file for reading text (line by line)
            with lzma.open(filename, "rt") as f:
                # skip the header
                next(f)
                if label in MODELS:
                    idx = MODELS.index(label)
                    prefix = y_line[:2 * idx]
                    suffix = y_line[2 * idx + 1:] if (2 * idx + 1) < len(y_line) else ''
                    y_line = f'{prefix}1{suffix}'
                for line in f:
                    line = line.strip()
                    if line:
                        yield f'{line},{y_line}'

    dataset = tf.data.Dataset.from_generator(lambda: read_xz_lines(filenames, labels, y_line), \
                                             output_types=tf.string, output_shapes=())


    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.AUTOTUNE)

    def scale(x, y):
        if scaler_x:
            mean_x, scale_x = tf.constant(scaler_x.mean_, dtype=tf.float32),  tf.constant(scaler_x.scale_, dtype=tf.float32)
            x = (x - mean_x) / scale_x
        return x, y

    dataset = dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = (
        dataset
        # .shuffle(buffer_size=10000)  # Adjust buffer_size as appropriate
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
        argparse.ArgumentParser(description="Train a BDCT model finder.")
    parser.add_argument('--train_data', nargs='+', type=str,
                        help="paths to the files where the encoded training data "
                             "for each model are stored (all data in one file must correspond to the same epi model)")
    parser.add_argument('--train_labels', nargs='+', type=str, choices=MODELS,
                        help="labels (epi model names) corresponding to the training data files (same order)")
    parser.add_argument('--val_data', nargs='+', type=str,
                        help="paths to the files where the encoded validation data "
                             "for each model are stored (all data in one file must correspond to the same epi model)")
    parser.add_argument('--val_labels', nargs='+', type=str, choices=MODELS,
                        help="labels (epi model names) corresponding to the validation data files (same order)")
    parser.add_argument('--model_path', required=False, default=MODEL_FINDER_PATH, type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path.")
    params = parser.parse_args()

    model_path = params.model_path
    os.makedirs(model_path, exist_ok=True)
    scaler_x = StandardScaler()

    x_indices, _, n_columns = get_data_characteristics(paths=params.train_data, scaler_x=scaler_x)

    ds_train = get_train_data(n_columns, x_indices, params.train_labels, filenames=params.train_data, \
                              scaler_x=scaler_x, batch_size=BATCH_SIZE, shuffle=False)
    ds_val = get_train_data(n_columns, x_indices, params.val_labels, filenames=params.val_data, \
                            scaler_x=scaler_x, batch_size=BATCH_SIZE, shuffle=False)


    model = build_model(n_x=len(x_indices), n_y=len(MODELS))

    # early stopping to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    # Training of the Network, with an independent validation set
    model.fit(ds_train, verbose=1, epochs=EPOCHS, validation_data=ds_val,
              callbacks=[early_stop])

    print(f'Saving the trained model to {model_path}...')

    save_model_keras(model, model_path)
    save_scaler_joblib(scaler_x, model_path, suffix='x')
    save_scaler_numpy(scaler_x, model_path, suffix='x')


if '__main__' == __name__:
    main()
