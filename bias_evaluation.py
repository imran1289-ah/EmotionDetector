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

# Import model definitions
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


def load_gender_data(transform):
    # Paths to gender-segmented data
    image_path_male = "bias_dataset/gender/male"
    image_path_female = "bias_dataset/gender/female"
    image_path_other = "bias_dataset/gender/other"

    # Load datasets
    dataset_male = ImageFolder(root=image_path_male, transform=transform)
    dataset_female = ImageFolder(root=image_path_female, transform=transform)
    dataset_other = ImageFolder(root=image_path_other, transform=transform)

    # Split datasets
    total_size_male = len(dataset_male)
    total_size_female = len(dataset_female)
    total_size_other = len(dataset_other)

    # 70-15-15 train-validation-test split as done previously
    train_size_male = int(0.7 * total_size_male)
    test_size_male = total_size_male - train_size_male

    train_size_female = int(0.7 * total_size_female)
    test_size_female = total_size_female - train_size_female

    train_size_other = int(0.7 * total_size_other)
    test_size_other = total_size_other - train_size_other

    _, test_set_male = random_split(dataset_male, [train_size_male, test_size_male])
    _, test_set_female = random_split(dataset_female, [train_size_female, test_size_female])
    _, test_set_other = random_split(dataset_other, [train_size_other, test_size_other])

    # Create DataLoader for test sets
    test_loader_male = DataLoader(test_set_male, batch_size=32, shuffle=False)
    test_loader_female = DataLoader(test_set_female, batch_size=32, shuffle=False)
    test_loader_other = DataLoader(test_set_other, batch_size=32, shuffle=False)

    return test_loader_male, test_loader_female, test_loader_other


def main():
    # Load and transform the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((256, 256)),
        ]
    )

    # Initialize dataset for each age group
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

    testLoaders = {}

    # Initialize models and set the device
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    all_metrics = {
        "Attribute": [],
        "Group": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
    }

    # Load gender data
    test_loader_male, test_loader_female, test_loader_other = load_gender_data(transform)

    testLoaders.update({
        "male": test_loader_male,
        "female": test_loader_female,
        "other": test_loader_other
    })

    # Add new metrics for each gender group to the dictionary
    for gender_group, loader in [('male', test_loader_male), ('female', test_loader_female),
                                 ('other', test_loader_other)]:
        model = CNN()
        model.to(device)
        model.load_state_dict(torch.load("emotion_classifier_model_cnn.pth", map_location=device))

        y_true, y_pred = evaluate_model(model, loader, device)
        accuracy, precision, recall, fscore = calculate_metrics(y_true, y_pred)

        all_metrics["Attribute"].append("gender")
        all_metrics["Group"].append(gender_group)
        all_metrics["Accuracy"].append(accuracy)
        all_metrics["Precision"].append(precision)
        all_metrics["Recall"].append(recall)
        all_metrics["F1-Score"].append(fscore)

    metrics_df1 = pd.DataFrame(all_metrics)
    temp_metrics_df1 = metrics_df1

    # Get the average for each metrics and assign as new row
    metrics_df1.loc[len(metrics_df1)] = {'Attribute': "", 'Group': "average", 'Accuracy': metrics_df1["Accuracy"].mean(),
                                       'Precision': metrics_df1["Precision"].mean(),
                                       "Recall": metrics_df1["Recall"].mean(), "F1-Score": metrics_df1["F1-Score"].mean()}


    testLoaders = {"young": test_loader_young, "middle-aged": test_loader_middle, "senior": test_loader_senior}

    all_metrics = {
        "Attribute": [],
        "Group": [],
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
    metrics_df2 = pd.DataFrame(all_metrics)
    temp_metrics_df2 = metrics_df2

    # Get the average for each metrics and assign as new row
    metrics_df2.loc[len(metrics_df2)] = {'Attribute': "", 'Group': "average", 'Accuracy': metrics_df2["Accuracy"].mean(),
                                       'Precision': metrics_df2["Precision"].mean(),
                                       "Recall": metrics_df2["Recall"].mean(), "F1-Score": metrics_df2["F1-Score"].mean()}

    # Print the DataFrame
    metrics_df = pd.concat([metrics_df2, metrics_df1])
    temp_metrics_df = pd.concat([temp_metrics_df2, temp_metrics_df1])

    metrics_df.loc[len(metrics_df)] = {'Attribute': "Overall average", 'Group': "", 'Accuracy': temp_metrics_df["Accuracy"].mean(),
                                       'Precision':temp_metrics_df["Precision"].mean(),
                                       "Recall": temp_metrics_df["Recall"].mean(), "F1-Score": temp_metrics_df["F1-Score"].mean()}

    print(metrics_df)


if __name__ == "__main__":
    main()
