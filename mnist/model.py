import tensorflow as tf

from gan import GAN


class MnistGAN(GAN):
    def __init__(self, input_shape):
        super(MnistGAN, self).__init__(input_shape=input_shape)

    def get_generator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=7 * 7 * 32,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Dropout(rate=.4),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=5,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=5,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.tanh)
        ])

    def get_discriminator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=.4),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
