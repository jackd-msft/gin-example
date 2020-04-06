import tensorflow as tf
import gin
from core_gin import Dense


@gin.configurable(blacklist=['inp'])
def custom_mlp(inp, dropout_rate=0.5):
    x = inp
    if inp.shape.ndims > 2:
        x = tf.keras.layers.Flatten()(x)
    for u in (1024, 16):
        x = Dense(u, activation='selu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x
