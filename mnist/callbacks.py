import os
import cv2
import numpy as np
import tensorflow as tf


def imshow_gan(batch, models):
    generator = models[0]
    if batch % 20 == 0:
        gan_output = np.asarray(generator(np.random.normal(size=(100, 100))))
        gan_output = gan_output.reshape((gan_output.shape[0],) + (28, 28))
        total_imgs = np.zeros(shape=(0, 800))

        for r in range(10):
            row_imgs = np.zeros(shape=(80, 0))
            for c in range(10):
                img = (gan_output[r * 10 + c] + 1) / 2
                img = cv2.copyMakeBorder(
                    src=cv2.resize(
                        src=img,
                        dsize=(60, 60)),
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


def imshow_cgan(batch, models):
    generator = models[0]
    if batch % 20 == 0:
        gan_output = np.asarray(generator([np.random.normal(size=(100, 100)), tf.keras.utils.to_categorical(np.random.randint(10, size=(100, 1)))]))
        gan_output = gan_output.reshape((gan_output.shape[0],) + (28, 28))
        total_imgs = np.zeros(shape=(0, 800))

        for r in range(10):
            row_imgs = np.zeros(shape=(80, 0))
            for c in range(10):
                img = (gan_output[r * 10 + c] + 1) / 2
                img = cv2.copyMakeBorder(
                    src=cv2.resize(
                        src=img,
                        dsize=(60, 60)),
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


def checkpoint(epoch, models):
    generator, discriminator = models
    if os.path.exists(f"./generator_epoch_{epoch - 1}.h5"):
        os.remove(f"./generator_epoch_{epoch - 1}.h5")
    if os.path.exists(f"./discriminator_epoch_{epoch - 1}.h5"):
        os.remove(f"./discriminator_epoch_{epoch - 1}.h5")
    generator.save(f"./generator_epoch_{epoch}.h5")
    discriminator.save(f"./discriminator_epoch_{epoch}.h5")
