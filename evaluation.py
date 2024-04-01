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
        micro_precision,
        micro_recall,
        micro_fscore,
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

    
    image_path = "dataset"  
    dataset = ImageFolder(root=image_path, transform=transform)

    # Calculate sizes for split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    validation_size = int(0.15 * total_size)
    test_size = total_size - (train_size + validation_size)

    # Set random state and split dataset
    torch.manual_seed(42)
    train_set, validation_set, test_set = random_split(
        dataset, [train_size, validation_size, test_size]
    )

    # Create a DataLoader for the testing set
    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize models and set the device
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    models = {
        "CNN": CNN(),
        "CNN Variant 2": CNNVariant2(),
        "CNN Variant 3": CNNVariant3(),
    }

    model_paths = {
        "CNN": "emotion_classifier_model_cnn.pth",
        "CNN Variant 2": "emotion_classifier_model_cnn_variant2.pth",
        "CNN Variant 3": "emotion_classifier_model_cnn_variant3.pth",
    }

    all_metrics = {
        "Model": [],
        "Macro P": [],
        "Macro R": [],
        "Macro F": [],
        "Micro P": [],
        "Micro R": [],
        "Micro F": [],
        "Accuracy": [],
    }

    for name, model in models.items():
        model.to(device)
        model.load_state_dict(torch.load(model_paths[name], map_location=device))

        y_true, y_pred = evaluate_model(model, test_loader, device)

        (
            precision,
            recall,
            fscore,
            micro_precision,
            micro_recall,
            micro_fscore,
            accuracy,
        ) = calculate_metrics(y_true, y_pred)

        # Add the metrics to the dictionary
        all_metrics["Model"].append(name)
        all_metrics["Macro P"].append(precision)
        all_metrics["Macro R"].append(recall)
        all_metrics["Macro F"].append(fscore)
        all_metrics["Micro P"].append(micro_precision)
        all_metrics["Micro R"].append(micro_recall)
        all_metrics["Micro F"].append(micro_fscore)
        all_metrics["Accuracy"].append(accuracy)

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True Class {i}" for i in range(len(cm))],
            columns=[f"Predicted Class {i}" for i in range(len(cm[0]))],
        )
        print("\nConfusion Matrix:")
        print(cm_df)
        print("\n")

    # Create a DataFrame with the collected metrics
    metrics_df = pd.DataFrame(all_metrics)

    # Set the model names as the index
    metrics_df.set_index("Model", inplace=True)

    # Print the DataFrame
    print(metrics_df)


if __name__ == "__main__":
    main()
