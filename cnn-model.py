from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolutional_layer = nn.Sequential(
            #First layer
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            
            

            #Second layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            

        )

        self.fullyconnected_layer = nn.Sequential(
            #Fully connected layer
            nn.Dropout(p=0.1),
            nn.Flatten()
            nn.Linear(32*64*64,64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fullyconnected_layer(x)
        return x
