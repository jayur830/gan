import tensorflow as tf

from gan import GAN


class Cifar10GAN(GAN):
    def get_generator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # (100,) -> (4 * 4 * 64,)
            tf.keras.layers.Dense(
                units=4 * 4 * 64,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (4 * 4 * 64,) -> (4, 4, 64)
            tf.keras.layers.Reshape(target_shape=(4, 4, 64)),
            tf.keras.layers.Dropout(rate=.2),
            # (4, 4, 64) -> (4, 4, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (4, 4, 32) -> (8, 8, 32)
            tf.keras.layers.UpSampling2D(),
            # (8, 8, 32) -> (8, 8, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (8, 8, 16) -> (16, 16, 16)
            tf.keras.layers.UpSampling2D(),
            # (16, 16, 16) -> (16, 16, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (16, 16, 8) -> (32, 32, 8)
            tf.keras.layers.UpSampling2D(),
            # (32, 32, 8) -> (32, 32, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=1,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])

    def get_discriminator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # (32, 32, 3) -> (32, 32, 8)
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (32, 32, 8) -> (16, 16, 8)
            tf.keras.layers.MaxPool2D(),
            # (16, 16, 8) -> (16, 16, 16)
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (16, 16, 16) -> (8, 8, 16)
            tf.keras.layers.MaxPool2D(),
            # (8, 8, 16) -> (8, 8, 32)
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (8, 8, 32) -> (4, 4, 32)
            tf.keras.layers.MaxPool2D(),
            # (4, 4, 32) -> (4, 4, 64)
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # Global Average Pooling
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dropout(rate=.2),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
