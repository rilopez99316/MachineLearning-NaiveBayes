import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.float32(x_train/255).reshape(x_train.shape[0],-1)
x_test = np.float32(x_test/255).reshape(x_test.shape[0],-1)

mean = np.mean(x_train)
x_train_binary = (x_train > mean).astype(int) #record boolean: as 1 or 0 and sets it to x_train_binary
x_test_binary = (x_test > mean).astype(int)#record boolean: as 1 or 0 and sets it to x_test_binary

print(x_train[0])

p_class = np.zeros(10)
for i in range(10):
  p_class[i] = len(y_train[y_train == i]) / x_train.shape[0]

print(p_class)

p_att_given_class = np.zeros((10,784)) #10 classes of 784 pixels
for i in range(10):
  index = y_train == i #traverses classes 
  p_att_given_class[i] = np.mean(x_train_binary[index], axis = 0) #calculates average of the 1's in each clss, for only column 0

# the number of times attribute j is equal to 1 in training instances of class j / over the number of training instances of class j
