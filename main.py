import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class GAN:
    def __init__(self,
                 kernel_initializer: str = "he_normal",
                 input_dim: int = 100):
        self.__input_dim = input_dim

        self.__generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=7 * 7 * 256,
                kernel_initializer=kernel_initializer,
                input_dim=self.__input_dim,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ThresholdedReLU(),
            tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ThresholdedReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ThresholdedReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.tanh)
        ])
        self.__generator.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=.001,
                momentum=.9,
                nesterov=False),
            loss=tf.keras.losses.binary_crossentropy)

        self.__discriminator = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ThresholdedReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=64,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ThresholdedReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=16,
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ThresholdedReLU(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
        self.__discriminator.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=.001,
                momentum=.9,
                nesterov=False),
            loss=tf.keras.losses.binary_crossentropy)
        self.__discriminator.trainable = False

        gan_input = tf.keras.layers.Input(shape=(self.__input_dim,))
        gan_output = self.__discriminator(self.__generator(gan_input))

        self.__gan = tf.keras.models.Model(gan_input, gan_output)
        self.__gan.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=.001,
                momentum=.9,
                nesterov=False),
            loss=tf.keras.losses.binary_crossentropy)

    def fit(self, x, epochs: int = 1, batch_size: int = 32):
        for i in range(epochs):
            print(f"{'-' * 10} {i + 1}/{epochs} Epochs {'-' * 10}")
            for j in tqdm(range(x.shape[0] // batch_size)):
                noise = np.random.normal(0, 1, size=(batch_size, self.__input_dim))

                y = np.zeros(2 * batch_size)
                y[:batch_size] = 1.

                d_input = np.concatenate([
                    x[np.random.randint(0, x.shape[0], size=batch_size)],
                    self.__generator.predict(noise)
                ])

                self.__discriminator.trainable = True
                self.__discriminator.train_on_batch(
                    d_input,
                    y)

                self.__discriminator.trainable = False
                self.__gan.train_on_batch(
                    np.random.normal(0, 1, size=(batch_size, self.__input_dim)),
                    np.ones(batch_size))

                if j == 1 or j % 5 == 0:
                    cv2.imshow("Test", cv2.resize(src=self.__generator.predict(noise[0].reshape((1,) + noise[0].shape)).reshape(28, 28, 1), dsize=(100, 100)))
                    cv2.waitKey(100)


if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()

    gan = GAN(input_dim=100)
    gan.fit(
        x=train_x.reshape(train_x.shape + (1,)) * 2. / 255. - 1.,
        epochs=10,
        batch_size=256)
