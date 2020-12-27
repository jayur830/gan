import tensorflow as tf

from cifar10.model import Cifar10GAN
from cifar10.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    print(train_x.shape)

    gan = Cifar10GAN(input_shape=(100,))
    gan.compile(
        discriminator_optimizer=tf.keras.optimizers.RMSprop(lr=1e-2, decay=3e-8),
        discriminator_loss=tf.losses.binary_crossentropy,
        gan_optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=.9),
        gan_loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=train_x / 255.,
        epochs=100,
        batch_size=256,
        callback=[
            imshow,
            checkpoint
        ])
