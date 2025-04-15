import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class MultiQuantilePinballLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, name="multi_quantile_pinball_loss"):
        super().__init__(name=name)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        """
        Compute the Pinball loss for multiple quantiles.

        Args:
        y_true: Tensor of true target values, shape (batch_size, n_targets).
        y_pred: Tensor of predicted values, shape (batch_size, n_targets * n_quantiles).

        Returns:
        A scalar tensor representing the aggregated Pinball loss.
        """

        n_quantiles = len(self.quantiles)
        n_targets = tf.shape(y_true)[1]

        # Reshape y_pred to (batch_size, n_targets, n_quantiles)
        y_pred = tf.reshape(y_pred, [-1, n_targets, n_quantiles])

        # Calculate the Pinball loss for each quantile
        loss = 0
        for i, tau in enumerate(self.quantiles):
            error = y_true - y_pred[:, :, i]  # Error for the i-th quantile
            quantile_loss = tf.maximum(tau * error, (tau - 1) * error)
            loss += tf.reduce_mean(quantile_loss)

        # Return the mean loss over all quantiles
        return loss / n_quantiles

    def get_config(self):
        # Serialize the quantiles
        config = super().get_config()
        config.update({
            "quantiles": self.quantiles
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the quantiles
        return cls(quantiles=config["quantiles"])