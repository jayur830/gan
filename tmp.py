import cv2
import numpy as np

if __name__ == '__main__':
    # img = np.random.randint(0, 256, size=(100, 100)).astype("float32")
    img = np.ones(shape=(100, 100)) * .5
    cv2.imshow("test", cv2.copyMakeBorder(
        src=img,
        top=10,
        left=10,
        bottom=10,
        right=10,
        borderType=cv2.BORDER_CONSTANT,
        value=1))
    cv2.waitKey()
