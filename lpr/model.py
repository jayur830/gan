import tensorflow as tf

from gan import GAN


class LprGAN(GAN):
    def __init__(self, input_dim):
        super(LprGAN, self).__init__(input_dim=input_dim)

    def get_generator(self):
        kernel_initializer = "he_normal"

        return tf.keras.models.Sequential([
            # (256,) -> (23 * 40 * 256)
            tf.keras.layers.Dense(
                units=23 * 40 * 256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (23 * 40 * 32) -> (23, 40, 32)
            tf.keras.layers.Reshape(target_shape=(23, 40, 256)),
            tf.keras.layers.Dropout(rate=.4),
            # (23, 40, 256) -> (46, 80, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (46, 80, 64) -> (92, 160, 64)
            tf.keras.layers.UpSampling2D(),
            # (92, 160, 64) -> (184, 320, 16)
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=.9),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (184, 320, 16) -> (368, 640, 16)
            tf.keras.layers.UpSampling2D(),
            # (368, 640, 16) -> (368, 640, 3)
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)
        ])

    def get_discriminator(self):
        kernel_initializer = "glorot_uniform"

        return tf.keras.models.Sequential([
            # (368, 640, 3) -> (184, 320, 8)
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (184, 320, 8) -> (92, 160, 16)
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=4,
                padding="same",
                strides=2,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=.1),
            # (92, 160, 16) -> (46, 80, 16)
            tf.keras.layers.MaxPool2D(),
            # (46, 80, 16) -> (23, 40, 32)
            tf.keras.layers.Conv2D(
                filters=32,
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
                activation="sigmoid",
                kernel_initializer=kernel_initializer)
        ])
