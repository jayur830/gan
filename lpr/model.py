import tensorflow as tf

from gan import GAN


class LprGAN(GAN):
    def get_generator(self, kernel_initializer: str = "glorot_uniform"):
        return tf.keras.models.Sequential([
            # (128,) -> (4 * 8 * 64,)
            tf.keras.layers.Dense(
                units=4 * 8 * 64,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (4 * 8 * 64,) -> (4, 8, 64)
            tf.keras.layers.Reshape(target_shape=(4, 8, 64)),
            tf.keras.layers.Dropout(rate=.3),
            # (4, 8, 64) -> (8, 16, 64)
            tf.keras.layers.UpSampling2D(),
            # (8, 16, 64) -> (10, 18, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (10, 18, 32) -> (20, 36, 32)
            tf.keras.layers.UpSampling2D(),
            # (20, 36, 32) -> (20, 36, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                padding="same",
                kernel_size=3,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (20, 36, 16) -> (23, 40, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=(4, 5),
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (23, 40, 8) -> (23, 40, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)
        ])

    def get_discriminator(self, kernel_initializer: str = "glorot_uniform"):
        return tf.keras.models.Sequential([
            # (23, 40, 3) -> (23, 40, 8)
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.Dropout(rate=.3),
            # Flatten
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
                kernel_initializer=kernel_initializer)
        ])
