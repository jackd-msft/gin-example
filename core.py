from typing import Tuple, Callable, Optional, Union
import tensorflow as tf
import tensorflow_datasets as tfds


def rescale_preprocess(image: tf.Tensor,
                       labels: tf.Tensor,
                       noise_stddev: Optional[float] = None
                       ) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255 * 2 - 1
    if noise_stddev is not None:
        image += tf.random.normal(shape=tf.shape(image), stddev=noise_stddev)
    return image, labels


def get_datasets(name: str = 'mnist',
                 batch_size: int = 16,
                 train_map_fn: Callable = rescale_preprocess,
                 test_map_fn: Callable = rescale_preprocess
                 ) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Get train/test datasets and number of classes for the named problem."""
    builder = tfds.builder(name)
    num_classes = builder.info.features[
        builder.info.supervised_keys[1]].num_classes
    train_ds, test_ds = builder.as_dataset(split=('train', 'test'),
                                           batch_size=batch_size,
                                           as_supervised=True)
    train_ds = train_ds.shuffle(1024).repeat()
    test_ds = test_ds.repeat()
    if train_map_fn is not None:
        train_ds = train_ds.map(train_map_fn)
    if test_map_fn is not None:
        test_ds = test_ds.map(test_map_fn)
    return train_ds, test_ds, num_classes


def cnn_features(image: tf.Tensor,
                 conv_filters=(16, 32),
                 dense_units=(256, ),
                 activation: Union[str, Callable] = 'relu',
                 Conv2D: Callable = tf.keras.layers.Conv2D,
                 Dense: Callable = tf.keras.layers.Dense) -> tf.Tensor:
    """Extract image features via CNN -> Flatten -> MLP."""
    x = image
    for u in conv_filters:
        x = Conv2D(u, 3, activation=activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = Dense(u, activation=activation)(x)
    return x


def mlp_features(inp: tf.Tensor,
                 dense_units=(512, 128),
                 activation='relu',
                 Dense: Callable = tf.keras.layers.Dense) -> tf.Tensor:
    """Extract features via Flatten -> MLP."""
    x = inp
    if inp.shape.ndims > 2:
        x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = Dense(u, activation=activation)(x)
    return x


def get_classification_model(
        input_spec: tf.TensorSpec,
        num_classes: int,
        features_fn: Callable = cnn_features,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'Adam',
        Dense: Callable = tf.keras.layers.Dense) -> tf.keras.Model:
    """
    Create and compile a keras model based on the input/outputs specs.
    
    Args:
        input_spec: tensor spec corresponding to the input.
        num_classes: number of classes in the output
        features_fn: function mapping batched inputs to rank 2 batched features
            of arbitrary size (this is passed into Dense to get logits).
            See `mlp_features` and `cnn_features` for examples.
        optimizer: optimizer to use in `tf.keras.Model.compile`.
        Dense: dense layer implementation.
    
    Returns:
        compiled keras model.
    """
    inp = tf.keras.Input(batch_shape=input_spec.shape, dtype=input_spec.dtype)
    features = features_fn(inp)
    logits = Dense(num_classes, activation=None)(features)
    model = tf.keras.Model(inp, logits)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def fit(model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        steps_per_epoch: int = 1000,
        validation_steps: int = 100,
        epochs: int = 2) -> None:
    """This wrapper around `model.fit`."""
    model.summary()
    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              validation_data=test_ds,
              epochs=epochs)
