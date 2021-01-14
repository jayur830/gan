import os
import cv2
import numpy as np


def imshow(batch, models):
    generator = models[0]
    if batch % 2 == 0:
        gan_output = generator.predict(np.random.normal(size=(1, 256)))
        cv2.imshow("Test", gan_output.reshape(gan_output.shape[1:]))
        cv2.waitKey(1)


def checkpoint(epoch, models):
    generator, discriminator = models
    if os.path.exists(f"./generator_epoch_{epoch - 1}.h5"):
        os.remove(f"./generator_epoch_{epoch - 1}.h5")
    if os.path.exists(f"./discriminator_epoch_{epoch - 1}.h5"):
        os.remove(f"./discriminator_epoch_{epoch - 1}.h5")
    generator.save(f"./generator_epoch_{epoch}.h5")
    discriminator.save(f"./discriminator_epoch_{epoch}.h5")
