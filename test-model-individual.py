from PIL import Image
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
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

#Loading custom image and transforming it
IMAGE_PATH = r"C:\Users\imran\Desktop\COMP 472\code\a2\cnn-train\dataset\Happy\00a7112212c133de419d2c89fd8be75729b041400d6458c3ed8db29a.jpg"
emotion_image = Image.open(IMAGE_PATH)
tensor_image = transform(emotion_image).unsqueeze(0)

#Predict the image by passing it to the model
model.eval()
with torch.no_grad():
        outputs = model(tensor_image)
        _, predicted = torch.max(outputs.data, 1)

labels = ["Focused", "Happy", "Neutral", "Suprised"]
predictedEmotion = labels[predicted.item()]

imageName = IMAGE_PATH.split("\\")[-1];

#Print the prediction
print(f"the emotion was found to be {predictedEmotion} for image {imageName}")

