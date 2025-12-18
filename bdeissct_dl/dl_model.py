import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable

from bdeissct_dl.bdeissct_model import F_E, F_S, UPSILON, F_S_X_S, UPS_X_C, REPRODUCTIVE_NUMBER, \
    INFECTION_DURATION, X_S, X_C, LA, PSI, RHO

LEARNING_RATE = 0.001

DELTA = 0.001
LOSS_WEIGHTS = {
    LA: 1,
    PSI: 1,
    REPRODUCTIVE_NUMBER: 1,
    INFECTION_DURATION: 1,
    UPS_X_C: 200,  # as there are 2 outputs, we multiply by 200 to scale it to [0, 200]
    F_S: 200, # as it is a value between 0 and 0.5, we multiply by 200 to scale it to [0, 100]
    F_E: 100,
    UPSILON: 100,
    F_S_X_S: 200,  # as there are 2 outputs, we multiply by 200 to scale it to [0, 200]
    X_C: 1,
    X_S: 1,
    RHO: 1,
}

QUANTILES = (0.5, )


@register_keras_serializable(package='bdeissct_dl', name='SSLayer')
class SSLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs shape: (batch, 2)
        f_S = half_sigmoid(inputs[:, 0:1])  # keepdims -> (batch, 1)
        X_S = relu_plus_one(inputs[:, 1:2])  # (batch, 1)
        return tf.concat([f_S, X_S], axis=-1)  # (batch, 2)

    def compute_output_shape(self, input_shape):
        # input_shape is (batch, 2) -> output_shape is (batch, 2)
        return input_shape[:-1] + (2,)

    def get_config(self):
        # If there are no special args, only return super() config
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package='bdeissct_dl', name='CTLayer')
class CTLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs shape: (batch, 2)
        ups = tf.sigmoid(inputs[:, 0:1])  # keepdims -> (batch, 1)
        X_C = relu_plus_one(inputs[:, 1:2])  # (batch, 1)
        return tf.concat([ups, X_C], axis=-1)  # (batch, 2)

    def compute_output_shape(self, input_shape):
        # input_shape is (batch, 2) -> output_shape is (batch, 2)
        return input_shape[:-1] + (2,)

    def get_config(self):
        # If there are no special args, only return super() config
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='bdeissct_dl', name='loss_ct')
def loss_ct(y_true, y_pred):

    # Unpack the true values
    p_true = y_true[:, 0]
    X_true = y_true[:, 1]

    # Unpack the predicted values
    p_pred = y_pred[:, 0]
    X_pred = y_pred[:, 1]

    # Relative error for X_C
    X_loss = tf.abs((X_pred - X_true) / X_true)

    # Absolute error for ups
    p_loss = tf.abs(p_pred - p_true)
    # p_loss = tf.abs((p_pred - p_true) / tf.maximum(p_true, 1e-2))

    mask = tf.cast(tf.greater(p_true, 1e-6), tf.float32)
    X_loss = tf.reduce_mean(mask * X_loss)

    # Combine the losses
    return tf.reduce_mean(X_loss + p_loss)

@tf.keras.utils.register_keras_serializable(package='bdeissct_dl', name='loss_ss')
def loss_ss(y_true, y_pred):

    # Unpack the true values
    p_true = y_true[:, 0]
    X_true = y_true[:, 1]

    # Unpack the predicted values
    p_pred = y_pred[:, 0]
    X_pred = y_pred[:, 1]

    # Relative error for X_S
    X_loss = tf.abs((X_pred - X_true) / X_true)

    # Absolute error for f_S, multiplied by 2, as f_S is in [0, 0.5]
    p_loss = 2 * tf.abs(p_pred - p_true)
    # p_loss = tf.abs((p_pred - p_true) / tf.maximum(p_true, 1e-2 / 2))

    mask = tf.cast(tf.greater(p_true, 1e-6), tf.float32)
    X_loss = tf.reduce_mean(mask * X_loss)

    # Combine the losses
    return tf.reduce_mean(X_loss + p_loss)


@tf.keras.utils.register_keras_serializable(package='bdeissct_dl', name='loss_prob')
def loss_prob(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_pred - y_true) / tf.maximum(y_true, 1e-2)))

@register_keras_serializable(package="bdeissct_dl", name="half_sigmoid")
def half_sigmoid(x):
    return 0.5 * tf.sigmoid(x)  # range ~ [0, 0.5)

@register_keras_serializable(package="bdeissct_dl", name="relu_plus_one")
def relu_plus_one(x):
    return 1 + tf.nn.relu(x)  # range ~ [1, infinity)



LOSS_FUNCTIONS = {
    LA: "mean_absolute_percentage_error",
    PSI: "mean_absolute_percentage_error",
    REPRODUCTIVE_NUMBER: "mean_absolute_percentage_error",
    INFECTION_DURATION: "mean_absolute_percentage_error",
    UPS_X_C: loss_ct,
    UPSILON: 'mae',
    RHO: 'mean_absolute_percentage_error',
    X_C: "mean_absolute_percentage_error",
    F_E: 'mae',
    F_S_X_S: loss_ss,
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
    # x = tf.keras.layers.Dropout(0.5, name='dropout3_50')(x)
    x = tf.keras.layers.Dense(16, activation='elu', name=f'layer4_dense32_elu')(x)

    outputs = {}

    if REPRODUCTIVE_NUMBER in target_columns:
        outputs[REPRODUCTIVE_NUMBER] = tf.keras.layers.Dense(1, activation="softplus", name=REPRODUCTIVE_NUMBER)(x) # positive values only
    if INFECTION_DURATION in target_columns:
        outputs[INFECTION_DURATION] = tf.keras.layers.Dense(1, activation="softplus", name=INFECTION_DURATION)(x) # positive values only
    if F_E in target_columns:
        outputs[F_E] = tf.keras.layers.Dense(1, activation="sigmoid", name=F_E)(x)
    # if F_S in target_columns:
    #     outputs[F_S_X_S] = SSLayer(name=F_S_X_S)(tf.keras.layers.Dense(2, activation=None, name="FS_XS_logits")(x))
    if F_S in target_columns:
        outputs[F_S] = tf.keras.layers.Dense(1, activation=half_sigmoid, name="FS_logits")(x)
    if X_S in target_columns:
        outputs[X_S] = tf.keras.layers.Dense(1, activation=relu_plus_one, name="XS_logits")(x)
    # if UPSILON in target_columns:
    #     outputs[UPS_X_C] = CTLayer(name=UPS_X_C)(tf.keras.layers.Dense(2, activation=None, name="ups_XC_logits")(x))
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
