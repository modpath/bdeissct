import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import LA, PSI, RHO
from bdeissct_dl.dl_model import build_model, LEARNING_RATE, LOSS_FUNCTIONS, LOSS_WEIGHTS
from bdeissct_dl.model_serializer import save_model_keras, save_scaler_joblib, save_scaler_numpy
from bdeissct_dl.scaler_fitting import fit_scalers
from bdeissct_dl.training import get_data_characteristics, get_train_data
from bdeissct_dl.bdeissct_model import CT_EPI_COLUMNS, CT_RATE_COLUMNS


def build_model(target_columns, n_x, optimizer=None, metrics=None):
    """
    Build a FFNN of funnel shape with 4 hidden layers.
    We use a 50% dropout after the first 2 hidden layers.
    This architecture follows the PhyloDeep paper [Voznica et al. Nature 2022].

    :param n_x: input size (number of features)
    :param optimizer: by default Adam with learning rate of 0.001
    :param metrics: evaluation metrics, by default no metrics
    :return: the model instance: tf.keras.models.Sequential
    """

    inputs = tf.keras.Input(shape=(n_x,))

    # (Your hidden layers go here)
    x = tf.keras.layers.Dense(8, activation='elu', name=f'layer1_dense8_elu')(inputs)
    x = tf.keras.layers.Dense(4, activation='elu', name=f'layer2_dense6_elu')(x)
    x = tf.keras.layers.Dense(4, activation='elu', name=f'layer2_dense4_elu')(x)
    x = tf.keras.layers.Dense(2, activation='elu', name=f'layer3_dense2_elu')(x)

    outputs = {}

    if LA in target_columns:
        outputs[LA] = tf.keras.layers.Dense(1, activation="softplus", name=LA)(x) # positive values only
    if PSI in target_columns:
        outputs[PSI] = tf.keras.layers.Dense(1, activation="softplus", name=PSI)(x) # positive values only
    if RHO in target_columns:
        outputs[RHO] = tf.keras.layers.Dense(1, activation="sigmoid", name=RHO)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                  loss={col: LOSS_FUNCTIONS[col] for col in outputs.keys()},
                  loss_weights={col: LOSS_WEIGHTS[col] for col in outputs.keys()},
                  metrics=metrics)
    return model

def main():
    """
    Entry point for DL model training with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Train -CT model rate parameter estimator from epi parameters.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--val_data', type=str, nargs='+',
                        help="path to the files where the encoded validation data are stored")

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs to train the model')
    parser.add_argument('--model_path', default=MODEL_PATH, type=str,
                        help="path to the folder where the trained model should be stored.")
    params = parser.parse_args()

    os.makedirs(params.model_path, exist_ok=True)

    # R,f_E,f_S,X_S,upsilon,X_C,kappa,la are given
    # psi is to be predicted

    feature_columns = CT_EPI_COLUMNS
    target_columns = CT_RATE_COLUMNS
    # reshuffle params.train_data order
    if len(params.train_data) > 1:
        np.random.shuffle(params.train_data)
    if len(params.val_data) > 1:
        np.random.shuffle(params.val_data)


    x_indices, y_col2index = get_data_characteristics(paths=params.train_data,
                                                      feature_columns=feature_columns,
                                                      target_columns=target_columns)

    scaler_x = StandardScaler()
    fit_scalers(paths=params.train_data, x_indices=x_indices, scaler_x=scaler_x)

    if scaler_x is not None:
        save_scaler_joblib(scaler_x, params.model_path, suffix='ct.x')
        save_scaler_numpy(scaler_x, params.model_path, suffix='ct.x')

    for col, y_idx in y_col2index.items():
        print(f'Training to predict {col} for CT...')

        model = build_model([col], n_x=len(x_indices))
        print(f'Building a model from scratch with {len(x_indices)} input features and {col} as output.')
        print(model.summary())

        ds_train = get_train_data([col], x_indices, [y_idx], file_pattern=None, filenames=params.train_data, \
                                  scaler_x=scaler_x, batch_size=BATCH_SIZE * 8, shuffle=True)
        ds_val = get_train_data([col], x_indices, [y_idx], file_pattern=None, filenames=params.val_data, \
                                scaler_x=scaler_x, batch_size=BATCH_SIZE * 8, shuffle=True)

        #early stopping to avoid overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

        #Training of the Network, with an independent validation set
        model.fit(ds_train, verbose=1, epochs=params.epochs, validation_data=ds_val, callbacks=[early_stop])

        print(f'Saving the trained model CT.{col} to {params.model_path}...')

        save_model_keras(model, path=params.model_path, model_name=f'CT.{col}')



if '__main__' == __name__:
    main()
