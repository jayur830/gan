import tensorflow as tf

from mnist.model import MnistGAN
from mnist.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()

    gan = MnistGAN(input_shape=(100,))
    gan.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=train_x.reshape(train_x.shape + (1,)) / 127.5 - 1.,
        epochs=100,
        batch_size=32,
        callback=[
            imshow,
            checkpoint
        ])
