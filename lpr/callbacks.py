import cv2
import numpy as np


def imshow(generator):
    gan_output = generator.predict(np.random.uniform(low=-1., size=(1, 128)))
    cv2.imshow("Test", gan_output.reshape(gan_output.shape[1:]))
    cv2.waitKey(1)


def checkpoint(generator):
    generator.save("generator.h5")
