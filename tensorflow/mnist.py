import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

(mnist_train, mnist_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


mnist_train = mnist_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
mnist_train = mnist_train.cache() \
    .shuffle(ds_info.splits['train'].num_examples) \
    .batch(128) \
    .prefetch(tf.data.experimental.AUTOTUNE)

mnist_test = mnist_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
mnist_test = mnist_test.batch(128) \
    .cache() \
    .prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=(2, 2), input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()

model.fit(mnist_train, epochs=6, validation_data=mnist_test)
