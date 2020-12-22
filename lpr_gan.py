import tensorflow as tf

from gan import GAN


class LprGAN(GAN):
    def get_generator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # (23 * 40 * 32,)
            tf.keras.layers.Dense(
                units=23 * 40 * 32,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=.01),
            # (23 * 40 * 32,) -> (23, 40, 32)
            tf.keras.layers.Reshape(target_shape=(23, 40, 32)),
            # (23, 40, 32) -> (23, 40, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=.01),
            # (23, 40, 16) -> (46, 80, 16)
            tf.keras.layers.UpSampling2D(),
            # (46, 80, 16) -> (46, 80, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=.01),
            # (46, 80, 8) -> (92, 160, 8)
            tf.keras.layers.UpSampling2D(),
            # (92, 160, 8) -> (92, 160, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                activation="sigmoid",
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer)
        ])

    def get_discriminator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # (92, 160, 16) -> (92, 160, 32)
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=.01),
            # (92, 160, 32) -> (46, 80, 32)
            tf.keras.layers.MaxPool2D(),
            # (46, 80, 32) -> (46, 80, 64)
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=.01),
            # (46, 80, 64) -> (23, 40, 64)
            tf.keras.layers.MaxPool2D(),
            # Flatten
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=.2),
            tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
                kernel_initializer=kernel_initializer)
        ])
