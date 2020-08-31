from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Input, InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# print('Trainning shape :', X_train.shape, Y_train.shape)
# print('Testing shape: ', X_test.shape, Y_test.shape)
# classes = np.unique(Y_train)
# nClass = len(classes)
# print("Total number of outputs: ", nClass)
# print("Output class", classes)
# plt.figure(figsize=[5,5])
# ##display the first image in the train data
# plt.plot(121)
# plt.imshow(X_train[0,:,:], cmap='gray')
# plt.title("Ground truth : {}".format(Y_train[0]))
# ##display the first image in the test data
# plt.plot(122)
# plt.imshow(X_test[0,:,:], cmap='gray')
# plt.title("Ground truth : {}".format(Y_test[0]))

##data preprocessing

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

## convert the training and testing label into one-hot endcoding vector

train_Y_one_hot = to_categorical(Y_train)
test_Y_one_hot = to_categorical(Y_test)

## split data (80% for training and 20% for testing)

X_train, X_valid, train_lable, valid_lable = train_test_split(X_train, train_Y_one_hot, test_size=0.2)

##build the network
batch_size = 64
epoch = 20
num_classes = 10

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
cnn_model.add(LeakyReLU(alpha=1))
cnn_model.add(MaxPooling2D((2, 2), padding='same'))
cnn_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='linear'))
cnn_model.add(LeakyReLU(alpha=0.1))
cnn_model.add(Dense(num_classes, activation='softmax'))
#complete the model
cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
cnn_model.summary()
##trainning the model
cnn_model.fit(X_train, train_lable, batch_size=batch_size, epochs=epoch, validation_data=(X_valid, valid_lable))

