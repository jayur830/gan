import numpy as np
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor


class CGAN(tf.keras.models.Model):
    def __init__(self, z_dim, y_dim, **kwargs):
        super(CGAN, self).__init__(**kwargs)
        self.__z_dim = z_dim
        self.__y_dim = y_dim
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
        self.compile_discriminator(optimizer, loss)
        self.compile_gan(optimizer, loss)

    def compile_discriminator(self, optimizer, loss):
        self.__discriminator.compile(
            optimizer=optimizer,
            loss=loss)
        self.__discriminator.trainable = False

    def compile_cgan(self, optimizer, loss):
        gan_input_z = tf.keras.layers.Input(shape=(self.__z_dim,))
        gan_input_y = tf.keras.layers.Input(shape=(self.__y_dim,))
        gan_output = self.__discriminator(self.__generator(tf.keras.layers.Concatenate()([gan_input_z, gan_input_y])))
        self.__cgan = tf.keras.models.Model(inputs=[gan_input_z, gan_input_y], outputs=gan_output)
        self.__cgan.compile(
            optimizer=optimizer,
            loss=loss)

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
        if y is None:
            raise ValueError("The argument y cannot be None.")

        lambda_callbacks = []
        if callbacks is not None:
            for callback in callbacks:
                if type(callback) is tf.keras.callbacks.LambdaCallback:
                    lambda_callbacks.append(callback)

        batch_count = x.shape[0] // batch_size

        callback_executor = ThreadPoolExecutor(max_workers=16)

        history = [[], []]

        for callback in lambda_callbacks:
            callback.on_train_begin([self.__generator, self.__discriminator])

        for epoch in range(1, epochs + 1):
            for callback in lambda_callbacks:
                callback.on_epoch_begin(epoch, [self.__generator, self.__discriminator])

            print(f"Epoch {epoch}/{epochs}")
            progbar = tf.keras.utils.Progbar(batch_count)
            avg_d_loss, avg_g_loss = 0, 0
            for batch in range(1, batch_count + 1):
                for callback in lambda_callbacks:
                    callback.on_batch_begin(batch, [self.__generator, self.__discriminator])

                z = np.random.normal(0, 1, size=(batch_size, self.__z_dim))
                y_fake = np.random.randint(y.shape[-1], size=(batch_size, 1))

                self.__discriminator.trainable = True
                d_loss = self.__discriminator.train_on_batch(
                    np.concatenate([
                        x[np.random.randint(0, x.shape[0], size=batch_size)],
                        self.__generator(np.concatenate([z, y_fake], axis=1))
                    ]),
                    np.concatenate([
                        np.ones(batch_size),
                        np.zeros(batch_size)
                    ])
                )

                z = np.random.normal(0, 1, size=(batch_size, self.__z_dim))
                y_gan = np.ones(batch_size)
                self.__discriminator.trainable = False
                g_loss = self.__cgan.train_on_batch(z, y_gan)

                avg_g_loss += g_loss
                avg_d_loss += d_loss

                progbar.update(current=batch, values=[
                    ("g_loss", g_loss if batch < batch_count else avg_g_loss / batch_count),
                    ("d_loss", d_loss if batch < batch_count else avg_d_loss / batch_count)
                ])

                for callback in lambda_callbacks:
                    callback.on_batch_end(batch, [self.__generator, self.__discriminator])
            history[0].append(avg_g_loss)
            history[1].append(avg_d_loss)

            for callback in lambda_callbacks:
                callback.on_epoch_end(epoch, [self.__generator, self.__discriminator])

            if callbacks is not None:
                futures = []
                for f in callbacks:
                    if type(f) is not tf.keras.callbacks.LambdaCallback:
                        futures.append(callback_executor.submit(f, epoch, [self.__generator, self.__discriminator]))
                for future in futures:
                    future.result()
        for callback in lambda_callbacks:
            callback.on_train_end([self.__generator, self.__discriminator])

        return history

    def get_generator(self) -> tf.keras.models.Model:
        pass

    def get_discriminator(self) -> tf.keras.models.Model:
        pass

    def load_generator(self, filepath):
        self.__generator = tf.keras.models.load_model(filepath=filepath, compile=False)

    def load_discriminator(self, filepath):
        self.__discriminator = tf.keras.models.load_model(filepath=filepath, compile=False)

    def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
        input_layer = tf.keras.layers.Input(shape=(self.__z_dim,))
        return tf.keras.models.Model(input_layer, self.__generator(input_layer)).predict(x)
