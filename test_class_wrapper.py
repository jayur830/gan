import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor


class GAN(tf.keras.models.Model):
    def __init__(self, input_shape, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.__input_shape = input_shape
        self.__generator = self.get_generator()
        self.__discriminator = self.get_discriminator()

    def get_config(self):
        return None

    def call(self, inputs, training=None, mask=None):
        return self.__generator(inputs)

    def compile(self,
              optimizer=None,
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
        self.__optimizer = optimizer
        self.__loss = loss

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.__generator.summary()
        self.__discriminator.summary()

    def fit(self,
          x=None,
          y=None,
          batch_size=128,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
        if x is None:
            raise ValueError("The argument x cannot be None.")
        batch_count = x.shape[0] // batch_size

        self.__discriminator.compile(
            optimizer=self.__optimizer,
            loss=self.__loss)
        self.__discriminator.trainable = False

        gan_input = tf.keras.layers.Input(shape=self.__input_shape)
        gan_output = self.__discriminator(self.__generator(gan_input))
        gan = tf.keras.models.Model(inputs=gan_input, outputs=gan_output)
        gan.compile(
            optimizer=self.__optimizer,
            loss=self.__loss)

        callback_executor = ThreadPoolExecutor(max_workers=16)

        history = [[], []]

        for e in range(1, epochs + 1):
            print(f"Epoch {e}/{epochs}")
            progbar = tf.keras.utils.Progbar(batch_count)
            avg_d_loss, avg_g_loss = 0, 0
            for i in range(batch_count):
                latent_var = np.random.normal(0, 1, size=(batch_size,) + self.__input_shape)

                image_batch = x[np.random.randint(0, x.shape[0], size=batch_size)]

                generated_images = self.__generator.predict(latent_var)
                x_discriminator = np.concatenate([image_batch, generated_images])

                y_discriminator = np.zeros(2 * batch_size)
                y_discriminator[:batch_size] = 0.9

                self.__discriminator.trainable = True
                d_loss = self.__discriminator.train_on_batch(x_discriminator, y_discriminator)

                latent_var = np.random.normal(0, 1, size=(batch_size,) + self.__input_shape)
                y_gan = np.ones(batch_size)
                self.__discriminator.trainable = False
                g_loss = gan.train_on_batch(latent_var, y_gan)

                avg_g_loss += g_loss
                avg_d_loss += d_loss

                progbar.update(current=i, values=[
                    ("g_loss", g_loss),
                    ("d_loss", d_loss)
                ])
            progbar.update(current=batch_count, values=[
                ("g_loss", avg_g_loss / batch_count),
                ("d_loss", avg_d_loss / batch_count)
            ])
            history[0].append(avg_g_loss)
            history[1].append(avg_d_loss)

            if callbacks is not None:
                futures = []
                for f in callbacks:
                    futures.append(callback_executor.submit(f, e, self.__generator, self.__input_shape))
                for future in futures:
                    future.result()

        return history

    def get_generator(self) -> tf.keras.models.Model:
        pass

    def get_discriminator(self) -> tf.keras.models.Model:
        pass


class MNISTGAN(GAN):
    def __init__(self, input_shape):
        super(MNISTGAN, self).__init__(input_shape=input_shape)

    def get_generator(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=256,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(units=512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(units=1024),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(units=784),
            tf.keras.layers.Activation(tf.keras.activations.tanh)
        ])

    def get_discriminator(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=1024,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                input_shape=(784,)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=512),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=256),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=1),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        ])


def plot_generated_images(epoch, generator, input_shape):
    examples = 100
    dim = (10, 10)
    figsize = (10, 10)
    noise = np.random.normal(0, 1, size=(examples,) + input_shape)
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)


if __name__ == '__main__':
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x = np.concatenate([x_train, x_test])

    gan = MNISTGAN(input_shape=(100,))
    gan.compile(
        optimizer=tf.optimizers.Adam(
            learning_rate=2e-4,
            beta_1=.5),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=x,
        epochs=400,
        batch_size=128,
        callbacks=[plot_generated_images])


