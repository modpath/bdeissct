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
    quantiles_tensor = tf.constant((0.025, 0.5, 0.975), dtype=tf.float32)
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
    x = inputs

    x = tf.keras.layers.Dense(64, activation='elu', name=f'layer1_dense64_elu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='elu', name=f'layer2_dense64_elu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(8, activation='elu', name=f'layer3_dense8_elu')(x)
    x = tf.keras.layers.Dense(8, activation='elu', name=f'layer4_dense8_elu')(x)
    return inputs, x


def get_outputs_pinball(target_columns, x, quantiles=(0.025, 0.5, 0.975)):
    outputs = {}
    for col in target_columns:
        outputs[col] = tf.keras.layers.Dense(len(quantiles), name=col)(x)
    return outputs
