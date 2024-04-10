import torch
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch.nn as nn

import time

# Import your model definitions
from cnn_model import CNN
from cnn_model2 import CNNVariant2
from cnn_model3 import CNNVariant3


def calculate_metrics(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    
    # Return metrics as a dictionary
    return {
        "Precision": precision,
        "Recall": recall,
        "F-Score": fscore,
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F-Score": micro_fscore,
        "Accuracy": accuracy,
    }


def evaluate_model(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(test_true_labels, test_predictions)
    return metrics



def k_fold_train_and_validate(model_class, dataset, k=10, num_epochs=10, batch_size=32, device='cpu'):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f"Starting fold {fold+1}")

        # Timing starts here
        fold_start_time = time.time()

        # Create SubsetRandomSamplers for training and validation datasets
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, test_ids)

        # Create data loaders for training and validation
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

        # Initialize model for the current fold
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop for the current fold
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Fold {fold+1}, Epoch {epoch+1}, Training loss: {total_loss/len(train_loader)}")
        
        # Validation loop for the current fold
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        
        # Timing ends here
        fold_end_time = time.time()
        fold_elapsed_time = fold_end_time - fold_start_time
        print(f"Fold {fold+1} completed in {fold_elapsed_time:.2f} seconds. Validation Accuracy: {accuracy}%")
        fold_metrics = evaluate_model(model, val_loader, device)
        fold_metrics['Fold'] = fold + 1 
        fold_results.append(fold_metrics)

    return fold_results

def main():
    # Load and transform the dataset
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Resize((256, 256)),
    ])

    image_path = "dataset"
    dataset = ImageFolder(root=image_path, transform=transform)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = {
        "CNN": CNN,
        "CNN Variant 2": CNNVariant2,
        "CNN Variant 3": CNNVariant3,
    }

    # Dictionary to collect metrics for all models
    all_metrics = {
        "Model": [],
        "Fold": [],
        "Accuracy": [],
        "Macro Precision": [],
        "Macro Recall": [],
        "Macro F-Score": [],
        "Micro Precision": [],
        "Micro Recall": [],
        "Micro F-Score": [],
    }

    for name, model_class in models.items():
        print(f"\nTraining and evaluating model: {name}")
        
        # Perform k-fold training and validation
        fold_metrics_list = k_fold_train_and_validate(model_class, dataset, k=10, num_epochs=10, batch_size=32, device=device)

        for fold_metrics in fold_metrics_list:
            all_metrics["Model"].append(name)
            all_metrics["Fold"].append(fold_metrics['Fold'])  # Assuming fold number is stored in fold_metrics
            all_metrics["Accuracy"].append(fold_metrics["Accuracy"])
            all_metrics["Macro Precision"].append(fold_metrics["Precision"])
            all_metrics["Macro Recall"].append(fold_metrics["Recall"])
            all_metrics["Macro F-Score"].append(fold_metrics["F-Score"])
            all_metrics["Micro Precision"].append(fold_metrics["Micro Precision"])
            all_metrics["Micro Recall"].append(fold_metrics["Micro Recall"])
            all_metrics["Micro F-Score"].append(fold_metrics["Micro F-Score"])
            

    # Convert collected metrics into a DataFrame for easier analysis and export
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df)

    # Calculate and print the average metrics for each model
    average_metrics_df = metrics_df.groupby("Model").mean().reset_index()
    print("\nAverage Metrics for Each Model Across All Folds:")
    print(average_metrics_df)

if __name__ == "__main__":
    main()