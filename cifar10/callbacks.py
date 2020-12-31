import os
import cv2
import numpy as np


def imshow(batch, generator):
    if batch % 2 == 0:
        gan_output = generator.predict(np.random.normal(size=(100, 100)))
        total_imgs = np.zeros(shape=(0, 800, 3))

        for r in range(10):
            row_imgs = np.zeros(shape=(80, 0, 3))
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
                    value=0)
                row_imgs = np.concatenate([row_imgs, img], axis=1)
            total_imgs = np.concatenate([total_imgs, row_imgs], axis=0)

        cv2.imshow("Test", total_imgs)
        cv2.waitKey(1)


def checkpoint(epoch, generator):
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    generator.save(f"./checkpoint/generator_epoch_{epoch}.h5")
