import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable

from bdeissct_dl.bdeissct_model import LA, PSI, F_E, F_S, X_S, UPSILON, X_C, PIS

# QUANTILES = (0.025, 0.5, 0.975)
QUANTILES = (0.5, )


@register_keras_serializable(package='bdeissct_dl', name='OutputTransformLayer')
class OutputTransformLayer(tf.keras.layers.Layer):

    def call(self, logits):

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

        X_S_out = 1 + tf.nn.softplus(X_S_logit) # X_S in [1, +inf)
        X_C_out = 1 + tf.nn.softplus(X_C_logit)

        f_E_out = tf.sigmoid(f_E_logit)  # f_E in [0, 1]
        f_S_out = 0.5 * tf.sigmoid(f_S_logit)  # f_S in [0, 0.5]
        ups_out = tf.sigmoid(ups_logit)

        pi_out = tf.nn.softmax(pi_logit, axis=-1)  # pi_* in [0, 1], sum to 1

        # Concatenate all outputs back together
        return tf.stack([la_out, psi_out,
                         f_E_out,
                         f_S_out, X_S_out,
                         ups_out, X_C_out,
                         pi_out[:, 0], pi_out[:, 1], pi_out[:, 2], pi_out[:, 3], pi_out[:, 4], pi_out[:, 5]
                         ], axis=1)

    def get_config(self):
        # If there are no special args, only return super() config
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable(package="bdeissct_dl", name="half_sigmoid")
def half_sigmoid(x):
    return 0.5 * tf.sigmoid(x)  # range ~ [0, 0.5)

@register_keras_serializable(package="bdeissct_dl", name="relu_plus_one")
def relu_plus_one(x):
    return 1 + tf.nn.relu(x)  # range ~ [1, infinity)


def build_model(n_x, n_y, optimizer=None, metrics=None, quantiles=QUANTILES):
    """
    Build a FFNN of funnel shape with 4 hidden layers.
    We use a 50% dropout after the first 2 hidden layers.
    This architecture follows the PhyloDeep paper [Voznica et al. Nature 2022].

    :param n_x: input size (number of features)
    :param n_y: output size (number of model parameters)
    :param optimizer: by default Adam with learning rate of 0.001
    :param metrics: evaluation metrics, by default no metrics
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

    la_out = tf.keras.layers.Dense(1, activation="softplus", name=LA)(x) # positive values only
    psi_out = tf.keras.layers.Dense(1, activation="softplus", name=PSI)(x) # positive values only

    f_E_out = tf.keras.layers.Dense(1, activation="sigmoid", name=F_E)(x)

    f_S_out = tf.keras.layers.Dense(1, activation=half_sigmoid, name=F_S)(x)
    X_S_out = tf.keras.layers.Dense(1, activation=relu_plus_one, name=X_S)(x)

    ups_out = tf.keras.layers.Dense(1, activation="sigmoid", name=UPSILON)(x)
    X_C_out = tf.keras.layers.Dense(1, activation=relu_plus_one, name=X_C)(x)

    pi_out = tf.keras.layers.Dense(6, activation="softmax", name=PIS)(x)  # pi_E, pi_I, pi_S, pi_EC, pi_IC, pi_SC

    outputs = {
        LA: la_out,
        PSI: psi_out,
        UPSILON: ups_out,
        X_C: X_C_out,
        F_E: f_E_out,
        F_S: f_S_out,
        X_S: X_S_out,
        PIS: pi_out
    }

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.summary()

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    model.compile(optimizer=optimizer,
                  loss={
                      LA: "mean_absolute_percentage_error",
                      PSI: "mean_absolute_percentage_error",
                      UPSILON: 'mae',
                      X_C: "mean_absolute_percentage_error",
                      F_E: 'mae',
                      F_S: 'mae',
                      X_S: "mean_absolute_percentage_error",
                      PIS: 'mae'
                  },
                  loss_weights={
                      LA: 1,
                      PSI: 1,
                      UPSILON: 100,
                      X_C: 1,
                      F_E: 100,
                      F_S: 200, # as it is within [0, 0.5], we multiply by 200 to scale it to [0, 100]
                      X_S: 1,
                      PIS: 600  # as pi_* are within [0, 1] each, we multiply by 600 to scale it to [0, 600]
                  },
                  metrics=metrics)
    return model
