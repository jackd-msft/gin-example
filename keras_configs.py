import tensorflow as tf
import gin

Adam = gin.external_configurable(tf.keras.optimizers.Adam)
SGD = gin.external_configurable(tf.keras.optimizers.SGD)

l2 = gin.external_configurable(tf.keras.regularizers.l2)
ExponentialDecay = gin.external_configurable(
    tf.keras.optimizers.schedules.ExponentialDecay)


@gin.configurable
def get_optimizer(optimizer_fn=Adam, learning_rate=1e-3):
    return optimizer_fn(learning_rate)
