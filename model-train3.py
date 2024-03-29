from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from cnn_model3 import CNNVariant3

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize the dataset
image_path = "dataset"
dataset = ImageFolder(root=image_path, transform=transform)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
validation_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - validation_size

torch.manual_seed(42)
train_set, validation_set, test_set = random_split(dataset, [train_size, validation_size, test_size])

# Initialize data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model = CNNVariant3()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set up the training process
num_epochs = 10
best_validation_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_training_loss = running_loss / len(train_loader)
    print(f'Training: Epoch {epoch + 1}/{num_epochs}, Loss: {avg_training_loss:.6f}')

    model.eval()
    validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_validation += (predicted == labels).sum().item()
            total_validation += labels.size(0)  # Correctly update total_validation

    avg_validation_loss = validation_loss / len(validation_loader)
    if total_validation > 0:  # Ensure we don't divide by zero
        validation_accuracy = 100 * correct_validation / total_validation
        print(
            f'Validation: Epoch {epoch + 1}/{num_epochs}, Loss: {avg_validation_loss:.6f}, Accuracy: {validation_accuracy:.2f}%')
    else:
        print(
            f'Validation: Epoch {epoch + 1}/{num_epochs}, Loss: {avg_validation_loss:.6f}, Accuracy: N/A - No validation data')

    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        torch.save(model.state_dict(), "emotion_classifier_model_cnn_variant3.pth")

# Test the model
test_correct = 0
test_total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')
