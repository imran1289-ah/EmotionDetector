from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from torchvision.datasets import ImageFolder
from cnn_model import CNN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((256,256)),
])

#Initializing dataset
image_path = "dataset"
dataset = ImageFolder(root=image_path, transform=transform)


#Splitting dataset to corresponding ratio
train_set = int(0.7 *len(dataset))
validation_set = int(0.15*len(dataset))
test_set = len(dataset) - train_set - validation_set

#Set random state and split dataset
torch.manual_seed(42)
train_set, validation_set, test_set = random_split(dataset, [train_set,validation_set, test_set])

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
displayedLossTraining = 0

for epoch in range(num_epochs):
    
    #Train the model on training set
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        #Passing model outputs and true labels to cross entropy function
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Perform backpropagation and optimized training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Show training loss
    displayedLossTraining += loss.item()

    #Display some metrics for training
    print(f'Training: Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss {displayedLossTraining}')

    #Initalize variable for validation
    BestLossValidation = 0
    totalValidation = 0
    correctValidation = 0
    displayedLossValidation = 0
    
    #Evaluate model on validation set
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(validation_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            totalValidation = totalValidation + labels.size(0)
            _, predicted = torch.max(outputs.data, 1)    
            correctValidation = correctValidation + (predicted == labels).sum().item()

    #Calculate metrics on validation set such as loss and accuracy
    displayedLossValidation = displayedLossValidation + loss.item()
    displayedAccuracy = (correctValidation/totalValidation)*100

   
    #Save model when there is validation loss and that there is a loss in validation instead of an increase at each epoch
    if (displayedLossValidation < BestLossValidation):
        #Save model
        BestLossValidation = displayedLossValidation
        torch.save(model.state_dict(), "emotion_classifier_model_cnn_bias.pth" )
    elif (BestLossValidation == 0):
        #Save the first epoch as the best model initially
        torch.save(model.state_dict(), "emotion_classifier_model_cnn_bias.pth" )
    
    #Display the metrics
    print(f'Validation: Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(validation_loader)}, Loss {displayedLossValidation}, Accuracy {displayedAccuracy}')


# After training and validation, during the testing phase:
model.eval()
test_predictions = []
test_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.numpy())
        test_true_labels.extend(labels.numpy())

# Convert lists to numpy arrays for metric calculation
test_predictions = np.array(test_predictions)
test_true_labels = np.array(test_true_labels)




