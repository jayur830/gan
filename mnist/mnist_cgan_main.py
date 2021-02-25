import tensorflow as tf
import numpy as np
import cv2

from mnist.mnist_cgan import MnistCGAN

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.reshape(train_x.shape + (1,)).astype("float32") / 255.
    test_x = test_x.reshape(test_x.shape + (1,)).astype("float32") / 255.
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)

    latent_dim = 100
    num_classes = train_y.shape[-1]
    epochs = 9
    batch_size = 128

    cgan = MnistCGAN(latent_dim, num_classes)
    cgan.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=1e-3, decay=3e-8),
        loss=tf.losses.binary_crossentropy)

    def on_batch_end(_, logs):
        outputs = np.asarray(cgan([
            np.random.normal(size=(100, latent_dim)),
            tf.keras.utils.to_categorical(np.random.randint(num_classes, size=(100, 1)), num_classes)
        ]))
        total_imgs = np.zeros(shape=(0, 500))

        for i in range(10):
            row_imgs = np.zeros(shape=(50, 0))
            for j in range(10):
                img = cv2.copyMakeBorder(
                    src=cv2.resize(
                        src=outputs[i * 10 + j],
                        dsize=(36, 36),
                        interpolation=cv2.INTER_AREA),
                    top=7,
                    bottom=7,
                    left=7,
                    right=7,
                    borderType=cv2.BORDER_CONSTANT,
                    value=1)
                row_imgs = np.concatenate([row_imgs, img], axis=1)
            total_imgs = np.concatenate([total_imgs, row_imgs], axis=0)
        cv2.imshow("Train", np.asarray(total_imgs))
        cv2.waitKey(1)

    history = cgan.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)])
    np.save(file="./history.npy", arr=np.asarray(history))
