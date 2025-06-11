import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable


# QUANTILES = (0.025, 0.5, 0.975)
QUANTILES = (0.5, )


@tf.keras.utils.register_keras_serializable(package='bdeissct_dl', name='RelAbsLoss')
class RelAbsLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
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
        la_loss = tf.abs((la_pred - la_true) / la_true)
        psi_loss = tf.abs((psi_pred - psi_true) / psi_true)
        X_S_loss = tf.abs((X_S_pred - X_S_true) / X_S_true)
        X_C_loss = tf.abs((X_C_pred - X_C_true) / X_C_true)

        # Absolute error for f_S, f_E, ups and pis
        f_S_loss = 2 * tf.abs(f_S_pred - f_S_true) # as f_S is in [0, 0.5], we multiply by 2 to scale it to [0, 1]
        f_E_loss = tf.abs(f_E_pred - f_E_true)
        ups_loss = tf.abs(ups_pred - ups_true)
        pi_E_loss = tf.abs(pi_E_pred - pi_E_true)
        pi_I_loss = tf.abs(pi_I_pred - pi_I_true)
        pi_S_loss = 2 * tf.abs(pi_S_pred - pi_S_true) # as f_S is in [0, 0.5], we multiply by 2 to scale it to [0, 1]
        pi_EC_loss = tf.abs(pi_EC_pred - pi_EC_true)
        pi_IC_loss = tf.abs(pi_IC_pred - pi_IC_true)
        pi_SC_loss = tf.abs(pi_SC_pred - pi_SC_true)

        # Combine the losses
        return tf.reduce_mean(la_loss + psi_loss + X_S_loss + X_C_loss + f_S_loss + f_E_loss + ups_loss
                              +  pi_E_loss + pi_I_loss + pi_S_loss + pi_EC_loss + pi_IC_loss + pi_SC_loss)

    def get_config(self):
        # Serialize the quantiles
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

        # def transform_to_one_inf(val):
        #     # output is in [1, +inf)
        #     return 1 + tf.nn.softplus(val)
        #
        # def transform_to_zero_half(val):
        #     # output is in [0, 0.5]
        #     return 0.5 * tf.sigmoid(val)
        #
        # def transform_to_fractions_of_one(val):
        #     # output is in [0, 1], sum to 1
        #     return tf.nn.softmax(val, axis=-1)

        X_S_out = 1 + tf.nn.softplus(X_S_logit) # X_S in [1, +inf)
        # X_S_out = tf.keras.layers.Lambda(transform_to_one_inf)(X_S_logit)  # X_S in [1, +inf)
        X_C_out = 1 + tf.nn.softplus(X_C_logit)
        # X_C_out = tf.keras.layers.Lambda(transform_to_one_inf)(X_C_logit)

        f_E_out = tf.sigmoid(f_E_logit)  # f_E in [0, 1]
        # f_E_out = tf.keras.layers.Lambda(tf.sigmoid)(f_E_logit)  # f_E in [0, 1]
        f_S_out = 0.5 * tf.sigmoid(f_S_logit)  # f_S in [0, 0.5]
        # f_S_out = tf.keras.layers.Lambda(transform_to_zero_half)(f_S_logit)  # f_S in [0, 0.5]
        ups_out = tf.sigmoid(ups_logit)
        # ups_out = tf.keras.layers.Lambda(tf.sigmoid)(ups_logit)

        pi_out = tf.nn.softmax(pi_logit, axis=-1)  # pi_* in [0, 1], sum to 1
        # pi_out = tf.keras.layers.Lambda(transform_to_fractions_of_one)(pi_logit)  # pi_* in [0, 1], sum to 1

        # print(pi_logit, pi_out)

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


def build_model(n_x, n_y=4, optimizer=None, loss=None, metrics=None, quantiles=QUANTILES):
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

    # logits = tf.keras.layers.Dense(n_out, activation=None)(x)

    la_out = tf.keras.layers.Dense(1, activation="softplus", name="la")(x) # positive values only
    psi_out = tf.keras.layers.Dense(1, activation="softplus", name="psi")(x) # positive values only

    f_E_out = tf.keras.layers.Dense(1, activation="sigmoid", name="f_E")(x)

    f_S_out = tf.keras.layers.Dense(1, activation=half_sigmoid, name="f_S")(x)

    X_S_out = tf.keras.layers.Dense(1, activation=relu_plus_one, name="X_S")(x)

    ups_out = tf.keras.layers.Dense(1, activation="sigmoid", name="ups")(x)
    X_C_out = tf.keras.layers.Dense(1, activation=relu_plus_one, name="X_C")(x)

    pi_out = tf.keras.layers.Dense(6, activation="softmax", name="pi")(x)  # pi_E, pi_I, pi_S, pi_EC, pi_IC, pi_SC



    # outputs = OutputTransformLayer()(logits)

    outputs = {
        "la": la_out,
        "psi": psi_out,
        "ups": ups_out,
        "X_C": X_C_out,
        "f_E": f_E_out,
        "f_S": f_S_out,
        "X_S": X_S_out,
        "pi": pi_out
    }

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
        # loss = MultiQuantilePinballLoss(quantiles)
        loss = RelAbsLoss()
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # if metrics is None:
    #     metrics = ['accuracy']

    model.compile(optimizer=optimizer,
                  loss={
                      "la": "mean_absolute_percentage_error",
                      "psi": "mean_absolute_percentage_error",
                      "ups": 'mae',
                      "X_C": "mean_absolute_percentage_error",
                      "f_E": 'mae',
                      "f_S": 'mae',
                      "X_S": "mean_absolute_percentage_error",
                      "pi": 'mae'
                  },
                  loss_weights={
                      "la": 1,
                      "psi": 1,
                      "ups": 100,
                      "X_C": 1,
                      "f_E": 100,
                      "f_S": 200, # as it is within [0, 0.5], we multiply by 200 to scale it to [0, 100]
                      "X_S": 1,
                      "pi": 600  # as pi_* are within [0, 1] each, we multiply by 600 to scale it to [0, 600]
                  },
                  metrics=metrics)
    return model
