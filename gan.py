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

        lambda_callbacks = []
        if callbacks is not None:
            for callback in callbacks:
                if type(callback) is tf.keras.callbacks.LambdaCallback:
                    lambda_callbacks.append(callback)

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

        for callback in lambda_callbacks:
            callback.on_train_begin(self.__generator)

        for epoch in range(1, epochs + 1):
            for callback in lambda_callbacks:
                callback.on_epoch_begin(epoch, self.__generator)

            print(f"Epoch {epoch}/{epochs}")
            progbar = tf.keras.utils.Progbar(batch_count)
            avg_d_loss, avg_g_loss = 0, 0
            for batch in range(1, batch_count + 1):
                for callback in lambda_callbacks:
                    callback.on_batch_begin(batch, self.__generator)

                latent_var = np.random.normal(0, 1, size=(batch_size,) + self.__input_shape)

                self.__discriminator.trainable = True
                d_loss = self.__discriminator.train_on_batch(
                    np.concatenate([
                        x[np.random.randint(0, x.shape[0], size=batch_size)],
                        self.__generator.predict(latent_var)
                    ]),
                    np.concatenate([
                        np.ones(batch_size),
                        np.zeros(batch_size)
                    ])
                )

                latent_var = np.random.normal(0, 1, size=(batch_size,) + self.__input_shape)
                y_gan = np.ones(batch_size)
                self.__discriminator.trainable = False
                g_loss = gan.train_on_batch(latent_var, y_gan)

                avg_g_loss += g_loss
                avg_d_loss += d_loss

                progbar.update(current=batch, values=[
                    ("g_loss", g_loss if batch < batch_count else avg_g_loss / batch_count),
                    ("d_loss", d_loss if batch < batch_count else avg_d_loss / batch_count)
                ])

                for callback in lambda_callbacks:
                    callback.on_batch_end(batch, self.__generator)
            history[0].append(avg_g_loss)
            history[1].append(avg_d_loss)

            for callback in lambda_callbacks:
                callback.on_epoch_end(epoch, self.__generator)

            if callbacks is not None:
                futures = []
                for f in callbacks:
                    if type(f) is not tf.keras.callbacks.LambdaCallback:
                        futures.append(callback_executor.submit(f, epoch, self.__generator))
                for future in futures:
                    future.result()
        for callback in lambda_callbacks:
            callback.on_train_end(self.__generator)

        return history

    def get_generator(self) -> tf.keras.models.Model:
        pass

    def get_discriminator(self) -> tf.keras.models.Model:
        pass
