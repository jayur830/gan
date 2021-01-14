import tensorflow as tf

from gan import GAN


class LPLocalGAN(GAN):
    def __init__(self, input_shape):
        super(LPLocalGAN, self).__init__(input_shape=input_shape)

    def get_generator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            # (100,) -> (256,)
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (256,) -> (3 * 6 * 256,)
            tf.keras.layers.Dense(
                units=3 * 6 * 256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (3 * 6 * 256,) -> (3, 6, 256)
            tf.keras.layers.Reshape(target_shape=(3, 6, 256)),
            tf.keras.layers.Dropout(rate=.4),
            # (3, 6, 256) -> (3, 6, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=5,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (3, 6, 64) -> (6, 12, 64)
            tf.keras.layers.UpSampling2D(),
            # (6, 12, 64) -> (6, 12, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=5,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (6, 12, 16) -> (12, 24, 16)
            tf.keras.layers.UpSampling2D(),
            # (12, 24, 16) -> (12, 24, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=5,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (12, 24, 8) -> (24, 48, 8)
            tf.keras.layers.UpSampling2D(),
            # (24, 48, 8) -> (24, 48, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(activation=tf.keras.activations.tanh)
        ])

    def get_discriminator(self):
        kernel_initializer = "glorot_uniform"

        return tf.keras.models.Sequential([
            # (24, 48, 3) -> (24, 48, 16)
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (24, 48, 16) -> (12, 24, 16)
            tf.keras.layers.MaxPool2D(),
            # (12, 24, 16) -> (12, 24, 64)
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
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
