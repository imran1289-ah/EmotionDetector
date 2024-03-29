import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
from cnn_model3 import CNNVariant3

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the dataset
image_path = "dataset"
full_dataset = ImageFolder(root=image_path, transform=transform)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(full_dataset))
validation_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

# Initialize the data loader for the test set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = CNNVariant3()
model.load_state_dict(torch.load("emotion_classifier_model_cnn_variant3.pth"))
model.eval()

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class to index mapping from the dataset
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

# Predict the test dataset
all_filenames = []
all_true_labels = []
all_predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Append data for filenames, true labels, and predictions
        all_filenames.extend([test_dataset.dataset.imgs[i][0] for i in labels])
        all_true_labels.extend(labels.cpu().numpy())
        all_predicted_labels.extend(preds.cpu().numpy())

# Mapping indices to class labels
all_true_labels = [idx_to_class[label] for label in all_true_labels]
all_predicted_labels = [idx_to_class[pred] for pred in all_predicted_labels]

correct = sum(p == l for p, l in zip(all_predicted_labels, all_true_labels))
accuracy = correct / len(all_true_labels)
print(f'Accuracy of the model on the complete dataset: {accuracy:.2f}')

# Saving the results to a CSV file
df = pd.DataFrame({
    'Filename': [os.path.basename(filename) for filename in all_filenames],
    'True Label': all_true_labels,
    'Predicted Label': all_predicted_labels
})

# Save the DataFrame to a CSV file
csv_file = 'test_set_predictions.csv'
df.to_csv(csv_file, index=False)
print(f"Predictions have been saved to {csv_file}")
