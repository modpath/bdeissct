from collections import defaultdict

import tensorflow as tf
from bdeissct_dl.bdeissct_model import REPRODUCTIVE_NUMBER, INFECTION_DURATION, INCUBATION_FRACTION, F_S, X_S, UPSILON, \
    X_C
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package="bdeissct_dl", name="half_sigmoid")
def half_sigmoid(x):
    return 0.5 * tf.sigmoid(x)  # range ~ [0, 0.5)

@register_keras_serializable(package="bdeissct_dl", name="relu_plus_one")
def relu_plus_one(x):
    return 1 + tf.nn.relu(x)  # range ~ [1, infinity)

@register_keras_serializable(package='bdeissct_dl', name='pinball_loss')
def pinball_loss(y_true, y_pred):
    """
    Create a pinball loss function for multiple quantiles.
    quantiles: tuple of quantile levels (e.g., (0.025, 0.5, 0.975))
    Assumes y_pred is a vector of quantile predictions, and y_true is the scalar true value.
    """
    quantiles_tensor = tf.constant((0.025, 0.5, 0.975), dtype=tf.float32)
    # y_true is scalar per sample, y_pred is vector of quantile preds
    y_true_expanded = tf.expand_dims(y_true, axis=-1)  # shape (batch, 1)
    errors = y_true_expanded - y_pred  # shape (batch, num_quantiles)
    loss_per_quantile = tf.maximum(quantiles_tensor * errors, (quantiles_tensor - 1) * errors)
    return tf.reduce_mean(loss_per_quantile)


LOSS_FUNCTIONS = {
    REPRODUCTIVE_NUMBER: "mean_absolute_percentage_error",
    INFECTION_DURATION: "mean_absolute_percentage_error",
    INCUBATION_FRACTION: "binary_crossentropy",
    UPSILON: 'binary_crossentropy',
    X_C: "mean_absolute_percentage_error",
    F_S: 'mse',
    X_S: "mean_absolute_percentage_error",
}

LOSS_WEIGHTS = {
    REPRODUCTIVE_NUMBER: 1,
    INFECTION_DURATION: 1,
    INCUBATION_FRACTION: 100,
    F_S: 100,
    UPSILON: 100,
    X_C: 1,
    X_S: 1
}

def get_model_layers(n_x):
    """
    Build a FFNN of funnel shape with 4 hidden layers.
    We use a 50% dropout after the first 2 hidden layers.
    This architecture follows the PhyloDeep paper [Voznica et al. Nature 2022].

    :param n_x: input size (number of features)
    :return: tuple (inputs, last_internal_layer)
    """

    inputs = tf.keras.Input(shape=(n_x,))
    x = inputs

    l2_reg = None

    # (Your hidden layers go here)
    # x = tf.keras.layers.Dense(128, activation='elu', name=f'layer1_dense128_elu', kernel_regularizer=l2_reg)(x)
    # # x = tf.keras.layers.Dropout(0.5, name='dropout1_50')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='elu', name=f'layer2_dense64_elu', kernel_regularizer=l2_reg)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='elu', name=f'layer3_dense64_elu', kernel_regularizer=l2_reg)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(8, activation='elu', name=f'layer4_dense8_elu', kernel_regularizer=l2_reg)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(8, activation='elu', name=f'layer5_dense8_elu', kernel_regularizer=l2_reg)(x)
    # x = tf.keras.layers.Dense(4, activation='elu', name=f'layer5_dense4_elu')(x)
    return inputs, x

def get_outputs_simple(target_columns, x):
    return {col: tf.keras.layers.Dense(1, name=col)(x) for col in target_columns}

def get_outputs_pinball(target_columns, x, quantiles=(0.025, 0.5, 0.975)):
    outputs = {}
    for col in target_columns:
        outputs[col] = tf.keras.layers.Dense(len(quantiles), name=col)(x)
    return outputs

def get_outputs(target_columns, x):
    outputs ={}
    if REPRODUCTIVE_NUMBER in target_columns:
        outputs[REPRODUCTIVE_NUMBER] = tf.keras.layers.Dense(1, activation="relu", name=REPRODUCTIVE_NUMBER)(
            x)  # positive values only
    if INFECTION_DURATION in target_columns:
        outputs[INFECTION_DURATION] = tf.keras.layers.Dense(1, activation="relu", name=INFECTION_DURATION)(
            x)  # positive values only
    if INCUBATION_FRACTION in target_columns:
        outputs[INCUBATION_FRACTION] = tf.keras.layers.Dense(1, activation="sigmoid", name=INCUBATION_FRACTION)(
            x)
    if F_S in target_columns:
        outputs[F_S] = tf.keras.layers.Dense(1, activation=half_sigmoid, name=F_S)(x)
    if X_S in target_columns:
        outputs[X_S] = tf.keras.layers.Dense(1, activation=relu_plus_one, name=X_S)(x)
    if UPSILON in target_columns:
        outputs[UPSILON] = tf.keras.layers.Dense(1, activation="sigmoid", name=UPSILON)(x)
    if X_C in target_columns:
        outputs[X_C] = tf.keras.layers.Dense(1, activation=relu_plus_one, name=X_C)(x)
    return outputs
