# gin-config example

[gin-config](https://github.com/google/gin-config) is a dependency injection framework developed by google (DeepMind?). This demo is intended to be a brief introduction to `gin` and show-case its potential.

In particular, it is NOT:
* a tutorial on deep learning, `tensorflow`, `keras` or `absl-py`;
* a thorough demonstration of all the glorious features `gin-config`;
* a demonstration of perfect experiment management; nor
* a substitute for the [user guide](https://github.com/google/gin-config/blob/master/docs/index.md).

Readers are assumed to have a working understanding of deep learning, `tf.keras` and basic command-line interfaces.

## Setup
```bash
pip install tensorflow           # nothing too compute-intensive, CPU fine
pip install gin-config absl-py tensorflow-datasets
git clone https://jackd-msft/gin-example.git
cd gin-example
```

## Core Baseline

Our baseline (non-gin) code is in `core.py`. It contains all relevant functions in a functional style. We'll focus on just the a couple of these:

```python
def cnn_features(image: tf.Tensor,
                 conv_filters=(16, 32),
                 dense_units=(256, ),
                 activation: Union[str, Callable] = 'relu',
                 Conv2D: Callable = tf.keras.layers.Conv2D,
                 Dense: Callable = tf.keras.layers.Dense) -> tf.Tensor:
    """Extract image features via CNN -> Flatten -> MLP."""
    ...


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
    ...
```

One only unusual aspect is the `Dense` and `Conv2D` arguments accepted. This is a general way of allowing the caller to specify things like regularizers, initializers and various other uniform hyperparameters used for layers in the model.

### CLI

The script for running experiments based on this model structure is `main_basic.py`, which uses a fairly standard CLI with `absl`. In terms of the model produced, it supports:
* a choice of Adam / SGD optimizers;
* a constant or exponentially decaying learning rate;
* a choice of two `feature_fn`s; and
* l2 weight decay.

For example, the code relevant to create the optimizer is as follows:

```python
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

...

lr = FLAGS.lr
steps = FLAGS.decay_steps
if steps is not None:
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, steps, FLAGS.decay_rate)
optimizer = _optimizer_fns[FLAGS.optimizer](lr)
```

while configuring `feature_fn`s involves:

```python
flags.DEFINE_string('features', default='cnn', help='feature extractor')
flags.DEFINE_float('weight_decay', default=None, help='l2 weight decay')

...

if FLAGS.features == 'mlp':
    features_fn = functools.partial(core.mlp_features, Dense=Dense)
elif FLAGS.features == 'cnn':
    features_fn = functools.partial(core.cnn_features,
                                    Dense=Dense,
                                    Conv2D=Conv2D)
else:
    raise ValueError(f'Invalid features {FLAGS.features}')
```

Usage would look something like
```bash
python main_basic.py --decay_steps=10000 --wd=4e-5
```

## gin-config version

Setup for our `gin-config` is in `core_gin.py` and is almost identical to `core.py`. Key differences are:
* `gin.external_configurable` wrappers on `tf.keras.layers.[Dense,Conv2D]`; and
* `@gin.configurable` annotations around functions we wish to configure.

```python
Dense = gin.external_configurable(tf.keras.layers.Dense)
Conv2D = gin.external_configurable(tf.keras.layers.Conv2D)

...

@gin.configurable(blacklist=['input_spec', 'num_classes'])
def get_classification_model(input_spec,
                             num_classes,
                             features_fn=cnn_features,
                             optimizer='Adam',
                             Dense=Dense):
    ...
```

This essentially means that all parameters apart from `input_spec` and `num_classes` can be configured ("injected") via `gin`, and that references to `Dense` / `Conv2D` (including the default value passed to `get_classification_model`) will also be able to be configured.

### CLI
The gin CLI is defined in `main_gin.py`. It is extremely basic.

```python
def parse_args():
    import sys
    args = sys.argv[1:]
    files = []
    bindings = []
    for arg in args:
        if '=' in arg:
            bindings.append(arg)
        else:
            files.append(arg if arg.endswith('.gin') else (arg + '.gin'))
    gin.parse_config_files_and_bindings(files, bindings)


def main():
    parse_args()
    train_ds, test_ds, num_classes = core.get_datasets()
    inputs_spec = train_ds.element_spec[0]
    model = core.get_classification_model(inputs_spec, num_classes)
    core.fit(model, train_ds, test_ds)
```

Essentially it takes command line args and interprets them as bindings if they contain an equals sign (`=`) or else a `gin` file (appending `.gin` if missing, and relative to the calling directory).

Critically, we also include some basic configuration files in the `configs` directory. An example call would be
```bash
python main_gin.py configs/mlp 'optimizer=@SGD()'
```

Of course this just hides most of the details inside `configs/mlp.gin`, so let's go through those lines one-by-one.

```gin
import core_gin
```

This ensures `core_gin` is imported and hence any `gin.configurable` or `gin.extenral_configurable` calls are made. Strictly speaking it isn't necessary since it will already be imported in `main_gin.py`, but including it in the config file means it'll still work even if using it from somewhere else that doesn't import it.

```gin
import keras_configs
```
`keras_configs.py` is another python file in this repository which makes various keras functions configurable using `gin.external_configurable`.

```gin
include 'configs/utils/reg.gin'
```
This is essentially a copy-paste of the contents. Note these are relative to the calling directory, not the location of the file containing it (see not below).

```gin
get_classification_model.features_fn = @mlp_features
```
This changes the default value of `get_classification_model` to the function `mlp_features`.

```gin
mlp_features.dense_units = %dense_units
get_classification_model.optimizer = %optimizer

dene_units = (128, 16)
optimizer = @Adam()    # requires `import keras_configs`
```
This changes the default value of `dense_units` and `optimizer` to the values in the corresponding macros, then defines the macro values. We could have got away with the simpler `get_classification_model.optimizer = @Adam()`, but then changing this in other config files or on the command line would have to be more verbose. For example, without the macro changing the optimizer to SGD would require:
```bash
python main_gin.py configs/mlp 'get_classification_model.optimizer=@SGD()'
```

`configs/utils` contains config files for other modifications including changing optimizers (`adam.gin` and `sgd.gin`), exponentially decaying learning rates (`lr_decay.gin`), adding l2 regularization (`reg.gin`) and using the same function with two different configurations for data augmentation during training (`augment.gin`).

See also `configs/custom_mlp.gin` for how to use a custom `features_fn` without touching `main_gin.py`.

## Wrapping existing code
As has been done with various `keras` functionality, we can wrap an entire interface using `gin` like is done in `core_wrapped.py`. Note that the original default `Dense` and `Conv2D` calls in `core.py` are from `tf.keras` rather than their counterparts wrapped in `gin.external_configurable`. Without updating those as is done in the `core_wrapped.config`, layers created with these unwrapped layers will not have default parameters overriden when using `configs/utils/reg.gin`.

### Pre-wrapped Keras
See [this repo](https://github.com/jackd/kblocks) for extensively wrapped keras. Example usage:

#### Setup
```bash
git clone https://github.com/jackd/kblocks.git
pip install -e kblocks
```

#### Usage
```gin
import kblocks.keras.optimizers
import kblocks.keras.regularizers
optimizer = @Adam()
Dense.kernel_regularizer = @l2()
```

## Using gin with a standard CLI
`main_gin.py` uses a simple API that redirects to `gin.parse_config_files_and_bindings`. For production or wherever it is desired to lock-down the API, it is straight-forward (albeit time-consuming) to write a standard CLI and parsing the result into a `gin` config.

```python
flags.DEFINE_float('lr', default=1e-4, help='learning rate')
flags.DEFINE_string('optimizer', default='Adam', help='optimizer')

FLAGS = flags.FLAGS

config = f'''
optimizer = @{FLAGS.optimizer}()
{FLAGS.optimizer}.learning_rate = {lr}
'''
gin.parse_bindings(config)
```

Alternatively, `gin.bind_parameter` could be used in a potentially more pythonic way.

## Pros, cons and pit-falls

Pros:
* configurability is defined alongside code and documentation
* easy to resue configs using `include`
* inject dependencies that `main` has no knowledge of using `import`
* simple logging of relevant config using `gin.operative_config_str()`

Cons:
* configs making extensive use of macros and `include`s can get confusing
* new dependency/syntax to learn
* stack traces can be annoying
* singletons are verbose
* no syntax highlighting/linting for `.gin` files

Pitfalls
* when using external configurables, make sure your code calls the wrapped versions. For example, the following will NOT have weight decay
```python
def mlp(x, units):
    for u in units:
        x = tf.keras.layers.Dense(u, activation='relu')
    return x

gin.external_configurable(tf.keras.layers.Dense)
gin.external_configurable(tf.keras.regularizers.l2)

config = '''
Dense.kernel_regularizer = @l2()
l2.l = 4e-5
'''
```
* `include`s are relative to calling directory, not where the including file is located. They also can't include environment variables or the home symbol (`~`) (see [this PR](https://github.com/google/gin-config/pull/25) for work-in-progress to allow this).
* There is a slight overhead for calling configured functions. If calling a method in a configured function in a tight loop you can avoid this with explicit currying
```python
# BAD VERSION - calls a configurable function 1000x
@gin.configurable(blacklist=['x'])
def my_bad_f(x, y=3):
    return x + y

x = 0
for i in range(1000):
    x = my_bad_f(x)

```

```python
# GOOD VERSION - calls a configurable function 1x
@gin.configurable
def get_good_f(y=3):
    def f(x):
        return x + y
    return f


f = get_good_f()
x = 0
for i in range(1000):
    x = f(x)
```
* `gin.operative_config_str()` won't report configuration of functions that are called for the first time after it. It is best called at the very end of a program, although it may be able to be called earlier, e.g. just be a `tf.keras.Model.fit`.
