import glob
import os
import lzma

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl import MODEL_PATH, BATCH_SIZE, EPOCHS
from bdeissct_dl.bdeissct_model import MODEL2TARGET_COLUMNS, QUANTILES, LA, PSI, UPSILON, X_C, KAPPA, F_E, F_S, \
    X_S, TARGET_COLUMNS_BDCT, PI_E, PI_I, PI_S, PI_IC, PI_SC, PI_EC
from bdeissct_dl.model_serializer import save_model_keras, save_scaler_joblib, save_scaler_numpy
from bdeissct_dl.pinball_loss import MultiQuantilePinballLoss
from bdeissct_dl.tree_encoder import SCALING_FACTOR, STATS


FEATURE_COLUMNS = [_ for _ in STATS if _ not in {'n_trees', 'n_tips', 'n_inodes', 'len_forest',
                                                 LA, PSI,
                                                 UPSILON, X_C, KAPPA,
                                                 F_E,
                                                 F_S, X_S,
                                                 PI_E, PI_I, PI_S,
                                                 PI_EC, PI_IC, PI_SC,
                                                 SCALING_FACTOR}]

def rel_abs_loss(y_true, y_pred):
    """
    Custom loss function for the BDEISSCT model.
    The predicted model parameters are assumed to be (in this order)
        ['la', 'psi',
        'f_E',
        'f_S', 'X_S',
        'upsilon', 'X_C',
        'pi_E', 'pi_I', 'pi_S', 'pi_E', 'pi_IC', 'pi_SC']

    The loss is calculated as follows:
    1. For la, psi, X_S and X_C: relative error is calculated as abs((pred - true) / (true + epsilon)),
       where epsilon is a small value to avoid division by zero.
    2. For f_E, f_S, upsilon and pi_*: absolute error is calculated as abs(pred - true).

    :param y_true: true values, shape [batch_size, 13]
    :param y_pred: predicted values, shape [batch_size, 13]
    :return: a scalar tensor representing the loss value
    """

    # Unpack the true values
    la_true = y_true[:, 0]
    psi_true = y_true[:, 1]

    f_E_true = y_true[:, 2]

    f_S_true = y_true[:, 3]
    X_S_true = y_true[:, 4]

    ups_true = y_true[:, 5]
    X_C_true = y_true[:, 6]

    pi_E_true = y_true[:, 7]
    pi_I_true = y_true[:, 8]
    pi_S_true = y_true[:, 9]
    pi_EC_true = y_true[:, 10]
    pi_IC_true = y_true[:, 11]
    pi_SC_true = y_true[:, 12]

    # Unpack the predicted values
    la_pred = y_pred[:, 0]
    psi_pred = y_pred[:, 1]

    f_E_pred = y_pred[:, 2]

    f_S_pred = y_pred[:, 3]
    X_S_pred = y_pred[:, 4]

    ups_pred = y_pred[:, 5]
    X_C_pred = y_pred[:, 6]

    pi_E_pred = y_pred[:, 7]
    pi_I_pred = y_pred[:, 8]
    pi_S_pred = y_pred[:, 9]
    pi_EC_pred = y_pred[:, 10]
    pi_IC_pred = y_pred[:, 11]
    pi_SC_pred = y_pred[:, 12]

    # Relative error for la, psi, X_S and X_C
    la_loss = tf.abs((la_pred - la_true) / (la_true + 1e-6))  # add small epsilon to avoid division by zero
    psi_loss = tf.abs((psi_pred - psi_true) / (psi_true + 1e-6))
    X_S_loss = tf.abs((X_S_pred - X_S_true) / (X_S_true + 1e-6))
    X_C_loss = tf.abs((X_C_pred - X_C_true) / (X_C_true + 1e-6))

    # Absolute error for f_S, f_E, ups and pis
    f_S_loss = tf.abs(f_S_pred - f_S_true)
    f_E_loss = tf.abs(f_E_pred - f_E_true)
    ups_loss = tf.abs(ups_pred - ups_true)
    pi_E_loss = tf.abs(pi_E_pred - pi_E_true)
    pi_I_loss = tf.abs(pi_I_pred - pi_I_true)
    pi_S_loss = tf.abs(pi_S_pred - pi_S_true)
    pi_EC_loss = tf.abs(pi_EC_pred - pi_EC_true)
    pi_IC_loss = tf.abs(pi_IC_pred - pi_IC_true)
    pi_SC_loss = tf.abs(pi_SC_pred - pi_SC_true)

    # Combine the losses
    return tf.reduce_mean(la_loss + psi_loss + X_S_loss + X_C_loss + f_S_loss + f_E_loss + ups_loss
                          +  pi_E_loss + pi_I_loss + pi_S_loss + pi_EC_loss + pi_IC_loss + pi_SC_loss)



def build_model(n_x, n_y=4, optimizer=None, loss=rel_abs_loss, metrics=None, quantiles=QUANTILES):
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

    n_q = len(quantiles)
    n_out = n_y * n_q

    inputs = tf.keras.Input(shape=(n_x,))

    # (Your hidden layers go here)
    x = tf.keras.layers.Dense(n_out << 4, activation='elu', name=f'layer1_dense{n_out << 4}_elu')(inputs)
    x = tf.keras.layers.Dropout(0.5, name='dropout1_50')(x)
    x = tf.keras.layers.Dense(n_out << 3, activation='elu', name=f'layer2_dense{n_out << 3}_elu')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout2_50')(x)
    x = tf.keras.layers.Dense(n_out << 2, activation='elu', name=f'layer3_dense{n_out << 2}_elu')(x)
    # x = tf.keras.layers.Dropout(0.5, name='dropout3_50')(x)
    x = tf.keras.layers.Dense(n_out << 1, activation='elu', name=f'layer4_dense{n_out << 1}_elu')(x)

    # This layer has 13 raw outputs ("logits"):
    #   [la_logit, psi_logit,
    #    f_E_logit,
    #    f_S_logit, X_S_logit,
    #    ups_logit, X_C_logit,
    #    pi_E_logit, pi_I_logit, pi_S_logit, pi_EC_logit, pi_IC_logit, pi_SC_logit]
    logits = tf.keras.layers.Dense(n_out, activation=None)(x)

    # Slice out each logit
    la_logit = logits[:, 0]
    psi_logit = logits[:, 1]

    f_E_logit = logits[:, 2]

    f_S_logit = logits[:, 3]
    X_S_logit = logits[:, 4]

    ups_logit = logits[:, 5]
    X_C_logit = logits[:, 6]

    pi_logit = logits[:, 7:]

    # Transform them into their desired ranges
    la_out = la_logit
    psi_out = psi_logit

    # X_S_out = 1 + tf.nn.softplus(X_S_logit) # X_S in [1, +inf)
    X_S_out = tf.keras.layers.Lambda(lambda _: 1 + tf.nn.softplus(_))(X_S_logit) # X_S in [1, +inf)
    # X_C_out = 1 + tf.nn.softplus(X_C_logit)
    X_C_out = tf.keras.layers.Lambda(lambda _: 1 + tf.nn.softplus(_))(X_C_logit)

    # f_E_out = tf.sigmoid(f_E_logit)  # f_E in [0, 1]
    f_E_out = tf.keras.layers.Lambda(tf.sigmoid)(f_E_logit) # f_E in [0, 1]
    # f_S_out = 0.5 * tf.sigmoid(f_S_logit)  # f_S in [0, 0.5]
    f_S_out = tf.keras.layers.Lambda(lambda _: 0.5 * tf.sigmoid(_))(f_S_logit)  # f_S in [0, 0.5]
    # ups_out = tf.sigmoid(ups_logit)
    ups_out = tf.keras.layers.Lambda(tf.sigmoid)(ups_logit)

    # pi_out = tf.nn.softmax(pi_logit, axis=-1)  # pi_* in [0, 1], sum to 1
    pi_out = tf.keras.layers.Lambda(lambda _: tf.nn.softmax(_, axis=-1))(pi_logit)  # pi_* in [0, 1], sum to 1

    print(pi_logit, pi_out)

    # Concatenate all outputs back together
    outputs = tf.keras.layers.Lambda(lambda _: tf.stack(_, axis=1))(
        [la_out, psi_out,
         f_E_out,
         f_S_out, X_S_out,
         ups_out, X_C_out,
         pi_out[:, 0], pi_out[:, 1], pi_out[:, 2],
         pi_out[:, 3], pi_out[:, 4], pi_out[:, 5]])

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)


    # model = tf.keras.models.Sequential(name="FFNN")
    # model.add(tf.keras.layers.InputLayer(shape=(n_x,), name='input_layer'))
    # model.add(tf.keras.layers.Dense(n_out << 4, activation='elu', name=f'layer1_dense{n_out << 4}_elu'))
    # model.add(tf.keras.layers.Dropout(0.5, name='dropout1_50'))
    # model.add(tf.keras.layers.Dense(n_out << 3, activation='elu', name=f'layer2_dense{n_out << 3}_elu'))
    # model.add(tf.keras.layers.Dropout(0.5, name='dropout2_50'))
    # model.add(tf.keras.layers.Dense(n_out << 2, activation='elu', name=f'layer3_dense{n_out << 2}_elu'))
    # # model.add(tf.keras.layers.Dropout(0.5, name='dropout3_50'))
    # model.add(tf.keras.layers.Dense(n_out << 1, activation='elu', name=f'layer4_dense{n_out << 1}_elu'))
    # model.add(tf.keras.layers.Dense(n_out, activation='linear', name=f'output_dense{n_out}_linear'))

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


def get_data_characteristics(paths, target_columns=TARGET_COLUMNS_BDCT, scaler_x=None, scaler_y=None):
    x_indices = []
    y_indices = []
    n_col = 0
    n_examples = 0

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
        n_examples += len(df)
        if scaler_x:
            X = df.iloc[:, x_indices].to_numpy(dtype=float, na_value=0)
            scaler_x.partial_fit(X)
        if scaler_y:
            Y = df.iloc[:, y_indices].to_numpy(dtype=float, na_value=0)
            scaler_y.partial_fit(Y)
    return x_indices, y_indices, n_col, n_examples

def get_train_data(n_input, columns_x, columns_y, file_pattern=None, filenames=None, scaler_x=None, scaler_y=None, \
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
        return X, Y


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
        if scaler_y:
            mean_y, scale_y = tf.constant(scaler_y.mean_, dtype=tf.float32), tf.constant(scaler_y.scale_, dtype=tf.float32)
            y = (y - mean_y) / scale_y
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
        argparse.ArgumentParser(description="Train a BDCT model.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/BD/{i}/trees.csv.xz' for i in range(10)],
                        help="path to the files where the encoded training data are stored")
    parser.add_argument('--val_data', type=str, nargs='+',
                        default=[f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/training/500_1000/BD/{i}/trees.csv.xz' for i in range(124, 128)],
                        help="path to the files where the encoded validation data are stored")
    parser.add_argument('--model_name', type=str,
                        # default='BD',
                        help="model name")
    parser.add_argument('--model_path', default=None, type=str,
                        help="path to the folder where the trained model should be stored. "
                             "The model will be stored at this path in the folder corresponding to the model name.")
    params = parser.parse_args()

    if params.model_path is None:
        model_path = MODEL_PATH.format(params.model_name)
    else:
        model_path = os.path.join(params.model_path, params.model_name)

    os.makedirs(model_path, exist_ok=True)
    # scaler_x, scaler_y = StandardScaler(), StandardScaler() #MinMaxScaler()
    scaler_x, scaler_y = StandardScaler(), None
    target_columns = MODEL2TARGET_COLUMNS[params.model_name]
    x_indices, y_indices, n_columns, n_examples = \
        get_data_characteristics(paths=params.train_data, target_columns=target_columns, \
                                 scaler_x=scaler_x, scaler_y=scaler_y)


    # reshuffle params.train_data order
    if len(params.train_data) > 1:
        np.random.shuffle(params.train_data)
    if len(params.val_data) > 1:
        np.random.shuffle(params.val_data)

    ds_train = get_train_data(n_columns, x_indices, y_indices, file_pattern=None, filenames=params.train_data, \
                              scaler_x=scaler_x, scaler_y=scaler_y, batch_size=BATCH_SIZE * 4, shuffle=False)
    ds_val = get_train_data(n_columns, x_indices, y_indices, file_pattern=None, filenames=params.val_data, \
                            scaler_x=scaler_x, scaler_y=scaler_y, batch_size=BATCH_SIZE * 4, shuffle=False)

    model = build_model(n_x=len(x_indices), n_y=len(y_indices), loss=rel_abs_loss)


    #early stopping to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    #Training of the Network, with an independent validation set
    model.fit(ds_train, verbose=1, epochs=EPOCHS, validation_data=ds_val,
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
