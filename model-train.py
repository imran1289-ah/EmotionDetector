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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((256,256)),
])

#Initializing dataset
image_path = "dataset/transformed_dataset"
dataset = ImageFolder(root=image_path, transform=transform)


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

#Initiazlie 10 epochs for training
num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()

        #Passing model outputs and true labels to cross entropy function
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Perform backpropagation and optimized training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()

        displayedEpoch =  (epoch+1)/(num_epochs)
        displayedStep = (i+1)/(len(train_loader))
        displayedLoss = loss.item()
        displayedAccuracy = (correct/total)*100

        print(f'Epoch {displayedEpoch}, Step {displayedStep}, Loss {displayedLoss}, Accuracy {displayedAccuracy}')

model.eval()
with torch.no_grad():
    test_correct = 0
    test_total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total = test_total + labels.size(0)
        test_correct = test_correct + (predicted == labels).sum().item()

        test_accuracy = test_correct/test_total
        print(f"Test Accuracy is {test_accuracy}")

#Save model
torch.save(model.state_dict(), "emotion_classifier_model_cnn.pth" )
        


