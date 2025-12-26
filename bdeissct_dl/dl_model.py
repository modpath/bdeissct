import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable

from bdeissct_dl.bdeissct_model import F_S, UPSILON, REPRODUCTIVE_NUMBER, \
    INFECTION_DURATION, X_S, X_C, RHO, INCUBATION_PERIOD

LEARNING_RATE = 0.001

LOSS_WEIGHTS = {
    REPRODUCTIVE_NUMBER: 1,
    INFECTION_DURATION: 1,
    INCUBATION_PERIOD: 1,
    F_S: 200, # as it is a value between 0 and 0.5, we multiply by 200 to scale it to [0, 100]
    UPSILON: 100,
    X_C: 1,
    X_S: 1
}

@register_keras_serializable(package="bdeissct_dl", name="half_sigmoid")
def half_sigmoid(x):
    return 0.5 * tf.sigmoid(x)  # range ~ [0, 0.5)

@register_keras_serializable(package="bdeissct_dl", name="relu_plus_one")
def relu_plus_one(x):
    return 1 + tf.nn.relu(x)  # range ~ [1, infinity)



LOSS_FUNCTIONS = {
    REPRODUCTIVE_NUMBER: "mean_absolute_percentage_error",
    INFECTION_DURATION: "mean_absolute_percentage_error",
    INCUBATION_PERIOD: "mae",
    UPSILON: 'mae',
    RHO: 'mean_absolute_percentage_error',
    X_C: "mean_absolute_percentage_error",
    F_S: 'mae',
    X_S: "mean_absolute_percentage_error",
}


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
    x = tf.keras.layers.Dense(128, activation='elu', name=f'layer1_dense256_elu')(inputs)
    x = tf.keras.layers.Dropout(0.5, name='dropout1_50')(x)
    x = tf.keras.layers.Dense(64, activation='elu', name=f'layer2_dense128_elu')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout2_50')(x)
    x = tf.keras.layers.Dense(32, activation='elu', name=f'layer3_dense64elu')(x)
    x = tf.keras.layers.Dense(16, activation='elu', name=f'layer4_dense32_elu')(x)

    outputs = {}

    if REPRODUCTIVE_NUMBER in target_columns:
        outputs[REPRODUCTIVE_NUMBER] = tf.keras.layers.Dense(1, activation="softplus", name=REPRODUCTIVE_NUMBER)(x) # positive values only
    if INFECTION_DURATION in target_columns:
        outputs[INFECTION_DURATION] = tf.keras.layers.Dense(1, activation="softplus", name=INFECTION_DURATION)(x) # positive values only
    if INCUBATION_PERIOD in target_columns:
        outputs[INCUBATION_PERIOD] = tf.keras.layers.Dense(1, activation="softplus", name=INCUBATION_PERIOD)(x) # positive values only
    if F_S in target_columns:
        outputs[F_S] = tf.keras.layers.Dense(1, activation=half_sigmoid, name="FS_logits")(x)
    if X_S in target_columns:
        outputs[X_S] = tf.keras.layers.Dense(1, activation=relu_plus_one, name="XS_logits")(x)
    if UPSILON in target_columns:
        outputs[UPSILON] = tf.keras.layers.Dense(1, activation="sigmoid", name="ups_logits")(x)
    if X_C in target_columns:
        outputs[X_C] = tf.keras.layers.Dense(1, activation=relu_plus_one, name="XC_logits")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                  loss={col: LOSS_FUNCTIONS[col] for col in outputs.keys()},
                  loss_weights={col: LOSS_WEIGHTS[col] for col in outputs.keys()},
                  metrics=metrics)
    return model
