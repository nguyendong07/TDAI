import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tranform

nomarlize = tranform.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_transform = tranform.Compose([tranform.RandomCrop(32, 4), tranform.ToTensor(), nomarlize])
aug_data = datasets.CIFAR10(root='C:/Users/ABC/Desktop/PetImages/Cat-Test/Dog', train=True, download=True, transform=train_transform)
