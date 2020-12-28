import tensorflow as tf

from gan import GAN


class MnistGAN(GAN):
    def get_generator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # (100,) -> (256,)
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (256,) -> (4 * 4 * 64,)
            tf.keras.layers.Dense(
                units=4 * 4 * 64,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (4 * 4 * 64,) -> (4, 4, 64)
            tf.keras.layers.Reshape(target_shape=(4, 4, 64)),
            tf.keras.layers.Dropout(rate=.4),
            # (4, 4, 64) -> (7, 7, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=
            )
            # (7, 7, 32) -> (10, 10, 16)
            # (10, 10, 16) -> (20, 20, 16)
            # (20, 20, 16) -> (24, 24, 8)
            # (24, 24, 8) -> (28, 28, 3)
            tf.keras.layers.Activation(tf.keras.activations.tanh)
        ])

    def get_discriminator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # Flatten
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=.3),
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
