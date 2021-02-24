import tensorflow as tf
import numpy as np


class ConditionalGAN(tf.keras.models.Model):
    def __init__(self, latent_dim, num_classes, **kwargs):
        super(ConditionalGAN, self).__init__(**kwargs)
        self._latent_dim = latent_dim
        self._num_classes = num_classes
        self.__generator = self.build_generator()
        self.__discriminator = self.build_discriminator()

    def get_config(self):
        return None

    def call(self, inputs, training=None, mask=None):
        return self.__generator(inputs)

    def compile(self,
              optimizer: object = 'rmsprop',
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

    def compile_gan(self, optimizer, loss):
        gan_input_z = tf.keras.layers.Input(shape=(self._latent_dim,))
        gan_input_y = tf.keras.layers.Input(shape=(self._num_classes,))
        gan_output = self.__discriminator([self.__generator([gan_input_z, gan_input_y]), gan_input_y])
        self.__gan = tf.keras.models.Model(
            inputs=[gan_input_z, gan_input_y],
            outputs=gan_output)
        self.__gan.compile(
            optimizer=optimizer,
            loss=loss)

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.__generator.summary()
        self.__discriminator.summary()

    def fit(self,
          x=None,
          y=None,
          batch_size=None,
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

        batch_count = x.shape[0] // batch_size

        history = [[], []]

        for callback in callbacks:
            callback.on_train_begin("")

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch + 1, "")

            print(f"\nEpoch {epoch + 1}/{epochs}")
            progbar = tf.keras.utils.Progbar(batch_count)
            avg_d_loss, avg_g_loss = 0, 0
            for batch_iter in range(batch_count):
                for callback in callbacks:
                    callback.on_batch_begin(None, "")

                z = np.random.normal(size=(batch_size, self._latent_dim))
                y_fake = tf.keras.utils.to_categorical(np.random.randint(self._num_classes, size=(batch_size, 1)), self._num_classes)

                self.__discriminator.trainable = True
                d_loss = self.__discriminator.train_on_batch(
                    x=[
                        np.concatenate([
                            np.asarray(self.__generator([z, y_fake])),
                            x[batch_iter * batch_size:(batch_iter + 1) * batch_size]
                        ], axis=0),
                        np.concatenate([
                            y_fake,
                            y[batch_iter * batch_size:(batch_iter + 1) * batch_size]
                        ], axis=0)
                    ],
                    y=np.concatenate([
                        np.zeros(shape=(batch_size, 1)),
                        np.ones(shape=(batch_size, 1))
                    ], axis=0))

                self.__discriminator.trainable = False
                g_loss = self.__gan.train_on_batch(
                    x=[z, y_fake],
                    y=np.ones(shape=(batch_size, 1)))

                avg_g_loss += g_loss
                avg_d_loss += d_loss

                progbar.update(current=batch_iter + 1, values=[
                    ("g_loss", g_loss if batch_iter < batch_count else avg_g_loss / batch_count),
                    ("d_loss", d_loss if batch_iter < batch_count else avg_d_loss / batch_count)
                ])

                for callback in callbacks:
                    callback.on_batch_end(None, "")
                history[0].append(avg_g_loss / batch_count)
                history[1].append(avg_d_loss / batch_count)
            for callback in callbacks:
                callback.on_epoch_end(epoch + 1, "")
        for callback in callbacks:
            callback.on_train_end("")

        return history

    def build_generator(self) -> tf.keras.models.Model:
        pass

    def build_discriminator(self) -> tf.keras.models.Model:
        pass

    def save_generator(self, filepath: str):
        self.__generator.save(filepath)

    def save_discriminator(self, filepath: str):
        self.__discriminator.save(filepath)

    def load_generator(self, filepath: str, compile: bool):
        self.__generator = tf.keras.models.load_model(filepath=filepath, compile=compile)

    def load_discriminator(self, filepath: str, compile: bool):
        self.__discriminator = tf.keras.models.load_model(filepath=filepath, compile=compile)

    @tf.function
    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        z = tf.keras.layers.Input(shape=(self._latent_dim,))
        y = tf.keras.layers.Input(shape=(self._num_classes,))
        return tf.keras.models.Model([z, y], self.__generator([z, y]))(x)