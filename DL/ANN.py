from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
# load the dataset
dataset = loadtxt('C:/Users/ABC/Desktop/New folder/TDAI/Data/bill_authentication.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:4]
y = dataset[:, 4]
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
print('Accuracy: %.2f' % (accuracy*100))
