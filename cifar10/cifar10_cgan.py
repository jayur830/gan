import tensorflow as tf

from cgan import ConditionalGAN


class Cifar10CGAN(ConditionalGAN):
    def build_generator(self) -> tf.keras.models.Model:
        kernel_initializer = "he_normal"
        bn_momentum = .9

        z = tf.keras.layers.Input(shape=(self._latent_dim,))
        y = tf.keras.layers.Input(shape=(self._num_classes,))

        g = tf.keras.layers.Concatenate()([z, y])

        # (latent_dim + num_classes,) -> (2 * 2 * 64,)
        g = tf.keras.layers.Dense(
            units=2 * 2 * 256,
            kernel_initializer=kernel_initializer)(g)
        # (2 * 2 * 64,) -> (2, 2, 64)
        g = tf.keras.layers.Reshape(target_shape=(2, 2, 256))(g)
        g = tf.keras.layers.Dropout(rate=.4)(g)
        g = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(g)
        g = tf.keras.layers.ReLU()(g)
        # (2, 2, 64) -> (4, 4, 32)
        g = tf.keras.layers.Conv2DTranspose(
            filters=256,
            kernel_size=5,
            padding="same",
            strides=2,
            kernel_initializer=kernel_initializer,
            use_bias=False)(g)
        g = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(g)
        g = tf.keras.layers.ReLU()(g)
        # (4, 4, 32) -> (8, 8, 16)
        g = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=5,
            padding="same",
            strides=2,
            kernel_initializer=kernel_initializer,
            use_bias=False)(g)
        g = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(g)
        g = tf.keras.layers.ReLU()(g)
        # (8, 8, 16) -> (16, 16, 8)
        g = tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=5,
            padding="same",
            strides=2,
            kernel_initializer=kernel_initializer,
            use_bias=False)(g)
        g = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(g)
        g = tf.keras.layers.ReLU()(g)
        # (16, 16, 8) -> (32, 32, 3)
        g = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=5,
            padding="same",
            strides=2,
            kernel_initializer=kernel_initializer)(g)
        g = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(g)

        return tf.keras.models.Model([z, y], g)

    def build_discriminator(self) -> tf.keras.models.Model:
        kernel_initializer = "he_normal"

        x = tf.keras.layers.Input(shape=(32, 32, 3))
        y = tf.keras.layers.Input(shape=(self._num_classes,))
        d = tf.keras.layers.Dense(
            units=32 * 32 * 1,
            kernel_initializer=kernel_initializer)(y)
        d = tf.keras.layers.Reshape(target_shape=(32, 32, 1))(d)

        d = tf.keras.layers.Concatenate()([x, d])
        # (32, 32, 4) -> (16, 16, 8)
        d = tf.keras.layers.SeparableConv2D(
            filters=8,
            kernel_size=3,
            padding="same",
            strides=2,
            kernel_initializer=kernel_initializer)(d)
        d = tf.keras.layers.LeakyReLU(alpha=.4)(d)
        # Global Average Pooling
        d = tf.keras.layers.GlobalAvgPool2D()(d)
        d = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=kernel_initializer)(d)
        d = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(d)

        return tf.keras.models.Model([x, y], d)
