import tensorflow as tf

from gan import GAN


class MnistGAN(GAN):
    def get_generator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # Latent variable
            tf.keras.layers.Dense(
                units=7 * 7 * 128,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.Reshape(target_shape=(7, 7, 128)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid),
        ])

    def get_discriminator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # (28, 28, 1) -> (28, 28, 8)
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # (28, 28, 8) -> (14, 14, 8)
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=.3),
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=.3),
            tf.keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=.3),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
