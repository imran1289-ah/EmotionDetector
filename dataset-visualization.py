import os
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader






# Define a transform to read the images
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def load_dataset(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset


def plot_class_distribution(dataset):
    classes = dataset.classes
    count = [0] * len(classes)
    for _, label in dataset:
        count[label] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(classes, count, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.show()


def display_sample_images(dataset):
    for class_index, class_name in enumerate(dataset.classes):
        # Generate indices for random images from the current class
        indices = np.where(np.array(dataset.targets) == class_index)[0]
        np.random.shuffle(indices)
        indices = indices[:25]  # Select 25 random images

        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.1)
        fig.suptitle(f'Class: {class_name}', fontsize=16)

        for ax in axes.flatten():
            ax.axis('off')

        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            ax = axes.flatten()[i]
            ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # Convert tensor to image format for displaying
            ax.set_title(class_name, fontsize=8)
        
        plt.show()  # Display the figure for the current class


def plot_pixel_intensity_distribution(dataset):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for class_index, class_name in enumerate(dataset.classes):
        indices = np.where(np.array(dataset.targets) == class_index)[0]
        np.random.shuffle(indices)
        indices = indices[:25]  # Use the same images as before.
        for idx in indices:
            img, _ = dataset[idx]
            img = np.transpose(img.numpy(), (1, 2, 0))
            for i, color in enumerate(['r', 'g', 'b']):
                axes[class_index].hist(img[:, :, i].ravel(), range=(0, 1), bins=256, color=color, alpha=0.5, label=color.upper() if idx == indices[0] else "")
            axes[class_index].set_title(class_name)
            axes[class_index].set_xlim([0, 1]) 
        if class_index == 0: 
            axes[class_index].legend()
    plt.show()


# Main function
if __name__ == '__main__':
    data_dir = './dataset'
    dataset = load_dataset(data_dir)
    plot_class_distribution(dataset)
    display_sample_images(dataset)
    plot_pixel_intensity_distribution(dataset)
    print('Done!')