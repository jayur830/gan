import tensorflow as tf

from mnist.model import MnistGAN
from mnist.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()

    gan = MnistGAN(input_shape=(100,))
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4, beta_1=.5),
        loss=tf.losses.binary_crossentropy)
    gan.fit_discriminator(
        x=train_x.reshape(train_x.shape + (1,)) / 255.,
        epochs=100,
        batch_size=32)
    # gan.fit_generator(
    #     x=train_x.reshape(train_x.shape + (1,)) / 255.,
    #     epochs=100,
    #     batch_size=256,
    #     callback=[
    #         imshow,
    #         checkpoint
    #     ])
