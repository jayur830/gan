import tensorflow as tf

from cgan import ConditionalGAN


class MnistCGAN(ConditionalGAN):
    def build_generator(self) -> tf.keras.models.Model:
        z = tf.keras.layers.Input(shape=(self._latent_dim,))
        y = tf.keras.layers.Input(shape=(self._num_classes,))

        g = tf.keras.layers.Concatenate()([z, y])
        g = tf.keras.layers.Dense(7 * 7 * 128)(g)
        g = tf.keras.layers.Reshape(target_shape=(7, 7, 128))(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.ReLU()(g)
        g = tf.keras.layers.Conv2DTranspose(64, 5, padding="same", strides=2)(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.ReLU()(g)
        g = tf.keras.layers.Conv2DTranspose(32, 5, padding="same", strides=2)(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.ReLU()(g)
        g = tf.keras.layers.Conv2DTranspose(1, 1)(g)
        g = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(g)

        return tf.keras.models.Model([z, y], g)

    def build_discriminator(self) -> tf.keras.models.Model:
        x = tf.keras.layers.Input(shape=(28, 28, 1))
        y = tf.keras.layers.Input(shape=(self._num_classes,))
        d = tf.keras.layers.Dense(7 * 7 * 16)(y)
        d = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(d)

        # Discriminator
        d = tf.keras.layers.Concatenate()([x, d])
        d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
        d = tf.keras.layers.Conv2D(32, 5, padding="same", strides=2)(d)
        d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
        d = tf.keras.layers.Conv2D(64, 5, padding="same", strides=2)(d)
        d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
        d = tf.keras.layers.Conv2D(128, 5, padding="same", strides=2)(d)
        d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
        d = tf.keras.layers.Conv2D(256, 5, padding="same", strides=2)(d)
        d = tf.keras.layers.Flatten()(d)
        d = tf.keras.layers.Dense(1)(d)
        d = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(d)

        return tf.keras.models.Model([x, y], d)