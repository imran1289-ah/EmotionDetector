from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset/emotiondata.csv")

X = df["ImagePath"].astype(str)
y = df["Label"].astype(str)

#Splitting dataset to 70/15/15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

