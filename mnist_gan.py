import tensorflow as tf

from gan import GAN


class MnistGAN(GAN):
    def get_generator(self, input_shape: tuple, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # Latent variable
            tf.keras.layers.Dense(
                units=7 * 7 * 8,
                kernel_initializer=kernel_initializer,
                use_bias=False,
                input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(
                units=7 * 7 * 64,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(
                units=7 * 7 * 256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (7 * 7 * 256,) -> (7, 7, 256)
            tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
            # (7, 7, 256) -> (7, 7, 128)
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (7, 7, 128) -> (7, 7, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (7, 7, 64) -> (14, 14, 64)
            tf.keras.layers.UpSampling2D(),
            # (14, 14, 64) -> (14, 14, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (14, 14, 32) -> (14, 14, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (14, 14, 16) -> (28, 28, 16)
            tf.keras.layers.UpSampling2D(),
            # (28, 28, 16) -> (28, 28, 8)
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (28, 28, 8) -> (28, 28, 1)
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])

    def get_discriminator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
