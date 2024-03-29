import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from cnn_model3 import CNNVariant3
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load dataset
image_path = r"dataset"
dataset = ImageFolder(root=image_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load the trained model
model = CNNVariant3()
model.load_state_dict(torch.load("emotion_classifier_model_cnn_variant3.pth"))
model.eval()

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class to index mapping from dataset
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# Predict the entire dataset
all_labels = []
all_predicted = []
with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(preds.cpu().numpy())

# Comparing the predictions with the actual labels
correct = sum(p == l for p, l in zip(all_predicted, all_labels))
accuracy = correct / len(all_labels)
print(f'Accuracy of the model on the complete dataset: {accuracy:.2f}')

# Mapping indices to classes
predicted_classes = [idx_to_class[pred] for pred in all_predicted]


# Saving the results to a CSV file
df = pd.DataFrame({
    'Filename': [os.path.basename(path) for path, _ in dataset.imgs],
    'True Label': [idx_to_class[label] for label in all_labels],
    'Predicted Label': predicted_classes
})

# Save the DataFrame to a CSV file
df.to_csv('dataset_predictions.csv', index=False)
print("Predictions have been saved to dataset_predictions.csv")
