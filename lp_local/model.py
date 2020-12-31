import tensorflow as tf

from gan import GAN


class LPLocalGAN(GAN):
    def __init__(self, input_shape):
        super(LPLocalGAN, self).__init__(input_shape=input_shape)

    def get_generator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            # (100,) -> (3 * 6 * 256,)
            tf.keras.layers.Dense(
                units=3 * 6 * 256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (3 * 6 * 256,) -> (3, 6, 256)
            tf.keras.layers.Reshape(target_shape=(3, 6, 256)),
            tf.keras.layers.Dropout(rate=.4),
            # (3, 6, 256) -> (6, 12, 128)
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (6, 12, 128) -> (12, 24, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (12, 24, 64) -> (24, 48, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (24, 48, 32) -> (48, 96, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (48, 96, 16) -> (96, 192, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (96, 192, 8) -> (96, 192, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        ])

    def get_discriminator(self):
        kernel_initializer = "glorot_uniform"

        return tf.keras.models.Sequential([
            # (96, 192, 3) -> (48, 96, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (48, 96, 8) -> (24, 48, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (24, 48, 16) -> (12, 24, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (12, 24, 32) -> (6, 12, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (6, 12, 64) -> (3, 6, 128)
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # Global Average Pooling
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dropout(rate=.4),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)
        ])
