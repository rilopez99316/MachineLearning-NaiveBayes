import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.float32(x_train/255).reshape(x_train.shape[0],-1)
x_test = np.float32(x_test/255).reshape(x_test.shape[0],-1)

mean = np.mean(x_train)
x_train_binary = (x_train > mean).astype(int) #record boolean: as 1 or 0 and sets it to x_train_binary
x_test_binary = (x_test > mean).astype(int)#record boolean: as 1 or 0 and sets it to x_test_binary

print(x_train[0])