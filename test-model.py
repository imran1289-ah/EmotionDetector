from PIL import Image
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

#Loading the trained model
model = CNN()
model.load_state_dict(torch.load("emotion_classifier_model_cnn.pth"), strict=False)

#Loading custom image and transforming it to tensor image
IMAGE_PATH = r"C:\Users\imran\Desktop\COMP 472\code\a2\COMP472\dataset\transformed_dataset\Neutral\cropped_emotions.278387f.png"
emotion_image = Image.open(IMAGE_PATH)
tensor_image = transform(emotion_image).unsqueeze(0).to(torch.device("cpu"))

#Predict the image
model.eval()
with torch.no_grad():
        outputs = model(tensor_image)
        _, predicted = torch.max(outputs.data, 1)

labels = ["Focused", "Happy", "Neutral", "Suprised"]
predictedEmotion = labels[predicted.item()]

#Print the prediction
print(f"image classified as {predictedEmotion}")

