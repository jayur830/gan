import tensorflow as tf

from gan import GAN


class MnistGAN(GAN):
    def get_generator(self, kernel_initializer: str = "he_normal"):
        return tf.keras.models.Sequential([
            # Latent variable
            tf.keras.layers.Dense(
                units=7 * 7 * 32,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.01),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=8,
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
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=.3),
            tf.keras.layers.Dense(
                units=32,
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
