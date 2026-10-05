import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable


QUANTILES = (0.025, 0.5, 0.975)

@register_keras_serializable(package='bdeissct_dl', name='pinball_loss')
def pinball_loss(y_true, y_pred):
    """
    Create a pinball loss function for multiple quantiles.
    quantiles: tuple of quantile levels (e.g., (0.025, 0.5, 0.975))
    Assumes y_pred is a vector of quantile predictions, and y_true is the scalar true value.
    """
    quantiles_tensor = tf.constant(QUANTILES, dtype=tf.float32)
    # y_true is scalar per sample, y_pred is vector of quantile preds
    y_true_expanded = tf.expand_dims(y_true, axis=-1)  # shape (batch, 1)
    errors = y_true_expanded - y_pred  # shape (batch, num_quantiles)
    loss_per_quantile = tf.maximum(quantiles_tensor * errors, (quantiles_tensor - 1) * errors)
    return tf.reduce_mean(loss_per_quantile)

def get_model_layers(n_x):
    """
    Build a FFNN of funnel shape with 4 hidden layers.
    We use a dropout after the first 2 hidden layers.

    :param n_x: input size (number of features)
    :return: tuple (inputs, last_internal_layer)
    """

    inputs = tf.keras.Input(shape=(n_x,))
    # 1. Normalize raw input features
    # (though they should already be normalized when calculating sum-stats)
    x = tf.keras.layers.Normalization(name="input_norm")(inputs)

    # 2. Block 1 (64 units + LayerNorm)
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='layer1_dense')(x)
    x = tf.keras.layers.LayerNormalization(name='layer1_norm')(x)
    x = tf.keras.layers.Activation('elu', name='layer1_elu')(x)

    # 3. Block 2 (64 units)
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='layer2_dense')(x)
    x = tf.keras.layers.LayerNormalization(name='layer2_norm')(x)
    x = tf.keras.layers.Activation('elu', name='layer2_elu')(x)

    # 4. Block 3 (32 units)
    x = tf.keras.layers.Dense(32, name='layer3_dense')(x)
    x = tf.keras.layers.LayerNormalization(name='layer3_norm')(x)
    x = tf.keras.layers.Activation('elu', name='layer3_elu')(x)

    return inputs, x


def get_outputs_pinball(target_columns, x, quantiles=QUANTILES):
    outputs = {}
    for col in target_columns:
        outputs[col] = tf.keras.layers.Dense(len(quantiles), name=col)(x)
    return outputs
