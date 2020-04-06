from absl import app, flags
import tensorflow as tf
import core
import functools

FLAGS = flags.FLAGS

_optimizer_fns = {
    'Adam': tf.keras.optimizers.Adam,
    'SGD': tf.keras.optimizers.SGD,
}

flags.DEFINE_string('optimizer', default='Adam', help='optimizer string')
flags.DEFINE_float('lr', default=1e-3, help='learning_rate')
flags.DEFINE_integer('decay_steps',
                     default=None,
                     help='number of steps per learning_rate decay')
flags.DEFINE_float('decay_rate', default=0.5, help='decay value.')
flags.DEFINE_float('wd', default=None, help='l2 weight decay')
flags.DEFINE_string('features', default='cnn', help='feature extractor')

flags.DEFINE_string('ds', default='mnist', help='dataset name')
flags.DEFINE_integer('batch_size', default=32, help='size of each batch')
flags.DEFINE_float('noise_stddev',
                   default=None,
                   help='stddev of noise in training')
flags.DEFINE_integer('steps_per_epoch',
                     default=100,
                     help='training steps per epoch')
flags.DEFINE_integer('validation_steps',
                     default=10,
                     help='validation steps per epoch')
flags.DEFINE_integer('epochs', default=2, help='number of epochs')


def main(_):
    import sys
    lr = FLAGS.lr
    steps = FLAGS.decay_steps
    if steps is not None:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            lr, steps, FLAGS.decay_rate)
    optimizer = _optimizer_fns[FLAGS.optimizer](lr)
    wd = FLAGS.wd

    Dense = tf.keras.layers.Dense
    Conv2D = tf.keras.layers.Conv2D
    if wd is not None:
        reg = tf.keras.regularizers.l2(wd)
        Dense = functools.partial(Dense, kernel_regularizer=reg)
        Conv2D = functools.partial(Conv2D, kernel_regularizer=reg)

    if FLAGS.features == 'mlp':
        features_fn = functools.partial(core.mlp_features, Dense=Dense)
    elif FLAGS.features == 'cnn':
        features_fn = functools.partial(core.cnn_features,
                                        Dense=Dense,
                                        Conv2D=Conv2D)
    else:
        raise ValueError(f'Invalid features {FLAGS.features}')

    train_map_fn = functools.partial(core.rescale_preprocess,
                                     noise_stddev=FLAGS.noise_stddev)
    train_ds, test_ds, num_classes = core.get_datasets(
        FLAGS.ds, FLAGS.batch_size, train_map_fn=train_map_fn)

    model = core.get_classification_model(train_ds.element_spec[0],
                                          num_classes, features_fn, optimizer,
                                          Dense)
    print(' '.join(sys.argv))  # basic logging of CL args
    core.fit(model, train_ds, test_ds, FLAGS.steps_per_epoch,
             FLAGS.validation_steps, FLAGS.epochs)


if __name__ == '__main__':
    app.run(main)
