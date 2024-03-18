from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageFolder
from cnn_model import CNN

#Initializing dataset
image_path = "dataset/transformed_dataset"
dataset = ImageFolder(root=image_path)

#Splitting dataset to train/test/validation
train_set = int(0.7 *len(dataset))
validation_set = int(0.15*len(dataset))
test_set = int(0.15*len(dataset))

train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [train_set,validation_set, test_set])

#Set data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False)

#Initalizing the custom model
model = CNN()

#Defining loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)