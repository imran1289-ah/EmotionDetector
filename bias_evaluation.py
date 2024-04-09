import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# Import your model definitions
from cnn_model import CNN
from cnn_model2 import CNNVariant2
from cnn_model3 import CNNVariant3


def calculate_metrics(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    accuracy = accuracy_score(y_true, y_pred)
    return (
        precision,
        recall,
        fscore,
        accuracy,
    )


def evaluate_model(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the same device as the model
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Move predictions and labels back to CPU for metrics calculation
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    return np.array(test_true_labels), np.array(test_predictions)


def main():
    # Load and transform the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((256, 256)),
        ]
    )

    
    #Initialize dataset for each age group
    image_path_middle = "bias_dataset/age/middle-aged"
    image_path_senior = "bias_dataset/age/senior"
    image_path_young = "bias_dataset/age/young"
      
    dataset_middle = ImageFolder(root=image_path_middle, transform=transform)
    dataset_senior = ImageFolder(root=image_path_senior, transform=transform)
    dataset_young = ImageFolder(root=image_path_young, transform=transform)

    # Calculate sizes for split for each aroup
    total_size_middle = len(dataset_middle)
    train_size_middle = int(0.7 * total_size_middle)
    validation_size_middle = int(0.15 * total_size_middle)
    test_size_middle = total_size_middle - (train_size_middle + validation_size_middle)

    total_size_senior = len(dataset_senior)
    train_size_senior = int(0.7 * total_size_senior)
    validation_size_senior = int(0.15 * total_size_senior)
    test_size_senior = total_size_senior - (train_size_senior + validation_size_senior)

    total_size_young = len(dataset_young)
    train_size_young = int(0.7 * total_size_young)
    validation_size_young = int(0.15 * total_size_young)
    test_size_young = total_size_young - (train_size_young + validation_size_young)



    # Set random state and split dataset for each group
    torch.manual_seed(42)
    train_set_middle, validation_set_middle, test_set_middle = random_split(
        dataset_middle, [train_size_middle, validation_size_middle, test_size_middle]
    )

    train_set_senior, validation_set_senior, test_set_senior = random_split(
        dataset_senior, [train_size_senior, validation_size_senior, test_size_senior]
    )

    train_set_young, validation_set_young, test_set_young = random_split(
        dataset_young, [train_size_young, validation_size_young, test_size_young]
    )

    # Create a DataLoader for the testing set
    test_loader_middle = DataLoader(
        test_set_middle, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader_senior = DataLoader(
        test_set_senior, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader_young = DataLoader(
        test_set_young, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    testLoaders = {"young":test_loader_young, "middle-aged":test_loader_middle, "senior":test_loader_senior}

    # Initialize models and set the device
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    
    all_metrics = {
        "Attribute":[],
        "Group":[],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
    }

    

    for age_group in testLoaders:

        model = CNN()
        model.to(device)
        model.load_state_dict(torch.load("emotion_classifier_model_cnn.pth", map_location=device))

        y_true, y_pred = evaluate_model(model, testLoaders[age_group], device)

        (
            accuracy,
            precision,
            recall,
            fscore,
        ) = calculate_metrics(y_true, y_pred)

        # Add the metrics to the dictionary for each age group
        all_metrics["Attribute"].append("age")
        all_metrics["Group"].append(age_group)
        all_metrics["Accuracy"].append(accuracy)
        all_metrics["Precision"].append(precision)
        all_metrics["Recall"].append(recall)
        all_metrics["F1-Score"].append(fscore)


    # Create a DataFrame with the collected metrics
    metrics_df = pd.DataFrame(all_metrics)

    #Get the average for each metrics and assign as new row
    metrics_df.loc[len(metrics_df)] = {'Attribute':"", 'Group':"average", 'Accuracy':metrics_df["Accuracy"].mean(), 'Precision':metrics_df["Precision"].mean(), "Recall":metrics_df["Recall"].mean(), "F1-Score":metrics_df["F1-Score"].mean()}

    # Print the DataFrame
    print(metrics_df)


if __name__ == "__main__":
    main()
