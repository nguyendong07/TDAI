import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import *
import os
from tqdm import tqdm
import random
datapath = "C:/Users/ABC/Desktop/PetImages"

Categories = ['Dog', 'Cat']

for category in Categories:
    path = os.path.join(datapath, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break #just want one for now so break
    break #...one and more

print(img_array)
print(img_array.shape)
new_size = 50
new_img = cv2.resize(img_array, (new_size, new_size))
plt.imshow(new_img, cmap='gray')
plt.show()

#create training data
training_data = []
def create_training_data():
    for category in Categories:
        path = os.path.join(datapath, category)
        class_num = Categories.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (new_size, new_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

# shuffle data
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])



