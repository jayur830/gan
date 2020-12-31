import os
import cv2
import numpy as np


def imshow(batch, generator):
    if batch % 2 == 0:
        gan_output = generator.predict(np.random.normal(size=(1, 256)))
        cv2.imshow("Test", gan_output.reshape(gan_output.shape[1:]))
        cv2.waitKey(1)


def checkpoint(epoch, generator):
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    generator.save(f"./checkpoint/generator_epoch_{epoch}.h5")
