import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    kernel_initializer = "he_normal"

    g_input_z = tf.keras.layers.Input(shape=(100,))
    g_input_y = tf.keras.layers.Input(shape=(10,))
    g = tf.keras.layers.Concatenate()([g_input_z, g_input_y])
    g = tf.keras.layers.Dense(
        units=7 * 7 * 128,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.Reshape(target_shape=(7, 7, 128))(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    g = tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    g = tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    g = tf.keras.layers.Conv2DTranspose(
        filters=32,
        kernel_size=5,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    g = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=5,
        kernel_initializer=kernel_initializer)(g)
    g = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(g)

    g = tf.keras.models.Model([g_input_z, g_input_y], g)

    d_input_x = tf.keras.layers.Input(shape=(28, 28, 1))
    d_input_y = tf.keras.layers.Dense(
        units=7 * 7 * 16,
        kernel_initializer=kernel_initializer,
        input_shape=(10,))
    d = tf.keras.layers.Concatenate()([d_input_x, d_input_y])
    d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
    d = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
    d = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
    d = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
    d = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=5,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.Flatten()(d)
    d = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(d)

    d = tf.keras.models.Model([d_input_x, d_input_y], d)

    d.trainable = False
    d.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)

    gan = tf.keras.models.Model(inputs=[g_input_z, g_input_y], outputs=d(g([g_input_z, g_input_y])))
    gan.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)


