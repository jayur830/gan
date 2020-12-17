import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class GAN:
    """
    Constructor
    :param kernel_initializer - Initializer for weights and biases
    :param input_dim - Dimension of the input
    """
    def __init__(self,
                 kernel_initializer: str = "he_normal",
                 input_dim: int = 100):
        # Input dimensions
        self.__input_dim = input_dim

        # Generator
        self.__generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=7 * 7 * 256,
                kernel_initializer=kernel_initializer,
                input_dim=self.__input_dim,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer,
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                padding="same",
                kernel_initializer=kernel_initializer),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])
        self.__generator.compile(
            optimizer=tf.keras.optimizers.Adam(lr=.0001, beta_1=.5),
            loss=tf.keras.losses.mean_squared_error)

        # Discriminator
        self.__discriminator = tf.keras.models.Sequential([
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
        self.__discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(lr=.0001, beta_1=.5),
            loss=tf.keras.losses.mean_squared_error)
        # Discriminator is not trainable yet.
        self.__discriminator.trainable = False

        # Create Generative Adversarial Networks
        gan_input = tf.keras.layers.Input(shape=(self.__input_dim,))
        gan_output = self.__discriminator(self.__generator(gan_input))

        self.__gan = tf.keras.models.Model(gan_input, gan_output)
        self.__gan.compile(
            optimizer=tf.keras.optimizers.Adam(lr=.0001, beta_1=.5),
            loss=tf.keras.losses.mean_squared_error)

    """
    :param x - Training dataset X
    :epochs - Training epochs
    :batch_size - Size of the batch in 1 iteration
    """
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
                self.__gan.train_on_batch(noise, np.ones(batch_size))

                if j == 1 or j % 5 == 0:
                    generated_imgs = self.__generator.predict(np.random.normal(0, 1, size=(100, self.__input_dim)))
                    total_imgs = np.zeros(shape=(0, 1000))

                    for r in range(10):
                        row_imgs = np.zeros(shape=(100, 0))
                        for c in range(10):
                            img = generated_imgs[r * 10 + c]
                            img = cv2.copyMakeBorder(
                                src=cv2.resize(
                                    src=img,
                                    dsize=(80, 80)),
                                top=10,
                                bottom=10,
                                left=10,
                                right=10,
                                borderType=cv2.BORDER_CONSTANT,
                                value=1)
                            row_imgs = np.concatenate([row_imgs, img], axis=1)
                        total_imgs = np.concatenate([total_imgs, row_imgs], axis=0)

                    cv2.imshow("Test", total_imgs)
                    cv2.waitKey(100)


if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()

    gan = GAN(input_dim=100)
    gan.fit(
        x=train_x.reshape(train_x.shape + (1,)) / 255.,
        epochs=10,
        batch_size=256)
