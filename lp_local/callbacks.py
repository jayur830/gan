import os
import cv2
import numpy as np

from glob import glob


def imshow(batch, models):
    generator = models[0]
    if batch % 2 == 0:
        gan_output = (generator.predict(np.random.normal(size=(1, 100))) + 1.) / 2.
        cv2.imshow("Test", gan_output.reshape(gan_output.shape[1:]))
        cv2.waitKey(1)


def checkpoint(epoch, models):
    generator, discriminator = models
    h5_list = glob("./*.h5")
    for filepath in h5_list:
        os.remove(filepath)
    generator.save(f"./generator_epoch_{epoch}.h5")
    discriminator.save(f"./discriminator_epoch_{epoch}.h5")
