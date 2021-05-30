import tensorflow as tf
import numpy as np
import cv2

from cifar10.cifar10_cgan import Cifar10CGAN

if __name__ == '__main__':
    (train_x, train_y), (_, _) = tf.keras.datasets.cifar10.load_data()
    print(train_x.shape)

    latent_dim = 100
    num_classes = train_y.shape[-1]
    epochs = 1000
    batch_size = 128

    cgan = Cifar10CGAN(latent_dim, num_classes)
    cgan.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3, decay=3e-8),
        loss=tf.losses.binary_crossentropy)

    def on_batch_end(_, logs):
        gan_output = np.asarray(cgan([
            np.random.normal(size=(100, latent_dim)),
            tf.keras.utils.to_categorical(np.random.randint(num_classes, size=(100, 1)), num_classes)
        ]))
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

    history = cgan.fit(
        x=(train_x / 255.).astype("float32"),
        y=train_y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)
        ])
    np.save(file="./history.npy", arr=np.asarray(history))
