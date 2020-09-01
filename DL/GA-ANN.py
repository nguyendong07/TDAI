import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def f(X):
    dataset = loadtxt('C:/Users/ABC/Desktop/New folder/TDAI/Data/pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=100, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

varbound = np.array([[0, 10]]*3)
print(varbound)
mode1 = ga(function=f, dimension=3, variable_type='real', variable_boundaries=varbound)

mode1.run()