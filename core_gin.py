import tensorflow as tf
import tensorflow_datasets as tfds
import gin

Dense = gin.external_configurable(tf.keras.layers.Dense)
Conv2D = gin.external_configurable(tf.keras.layers.Conv2D)


@gin.configurable(blacklist=['image', 'labels'])
def rescale_preprocess(image, labels, noise_stddev=None):
    image = tf.cast(image, tf.float32) / 255 * 2 - 1
    if noise_stddev is not None:
        image += tf.random.normal(shape=tf.shape(image), stddev=noise_stddev)
    return image, labels


@gin.configurable
def get_datasets(name='mnist',
                 batch_size=16,
                 train_map_fn=rescale_preprocess,
                 test_map_fn=rescale_preprocess):
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


@gin.configurable(blacklist=['image'])
def cnn_features(image,
                 conv_filters=(16, 32),
                 dense_units=(256, ),
                 activation='relu',
                 Conv2D=Conv2D,
                 Dense=Dense):
    x = image
    for u in conv_filters:
        x = Conv2D(u, 3, activation=activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = Dense(u, activation=activation)(x)
    return x


@gin.configurable(blacklist=['inp'])
def mlp_features(inp, dense_units=(512, 128), activation='relu', Dense=Dense):
    x = inp
    if inp.shape.ndims > 2:
        x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = Dense(u, activation=activation)(x)
    return x


@gin.configurable(blacklist=['input_spec', 'num_classes'])
def get_classification_model(input_spec,
                             num_classes,
                             features_fn=cnn_features,
                             optimizer='Adam',
                             Dense=Dense):
    inp = tf.keras.Input(batch_shape=input_spec.shape, dtype=input_spec.dtype)
    features = features_fn(inp)
    logits = Dense(num_classes, activation=None)(features)
    model = tf.keras.Model(inp, logits)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


@gin.configurable(blacklist=['model', 'train_ds', 'test_ds'])
def fit(model,
        train_ds,
        test_ds,
        steps_per_epoch=1000,
        validation_steps=100,
        epochs=2):
    model.summary()
    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              validation_data=test_ds,
              epochs=epochs)
