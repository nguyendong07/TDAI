import PIL
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = False
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.figsize'] = 15, 25

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
])

dataset = torchvision.datasets.ImageFolder('C:/Users/ABC/Desktop/New folder/TDAI/Data/DataSigned/20.signed', transform=transforms)
show_dataset(dataset)