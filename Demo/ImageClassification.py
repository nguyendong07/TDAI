#import lib
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
#create model
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=64, activation='relu'))

classifier.add(Dropout(0.5))

# output layer
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#split data
training_set = train_datagen.flow_from_directory('C:/Users/ABC/Desktop/PetImages/CatTrain',
                                                 target_size=(200, 200),
                                                 batch_size=32,
                                                 class_mode="binary")

test_set = test_datagen.flow_from_directory('C:/Users/ABC/Desktop/PetImages/CatTest',
                                            target_size=(200, 200),
                                            batch_size=32,
                                            class_mode="binary")

#set parameter for model
history = classifier.fit(training_set,
                         steps_per_epoch=100,
                         epochs=20,
                         validation_data=test_set,
                         validation_steps=200)

#test model
test_image = image.load_img('C:/Users/ABC/Desktop/PetImages/Dog/1.JPG', target_size=(200, 200))
#test_image.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'This is Cat'
else:
    prediction = 'This is not Cat'

print(prediction)
