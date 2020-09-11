import torch

FILE = "C:/Users/ABC/Desktop/New folder/TDAI/Demo/DetectObject/model.pth"
model = torch.load(FILE)
model.eval()