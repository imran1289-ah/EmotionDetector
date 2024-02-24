import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm
from multiprocessing import Pool


# Define a transform to read the images
transform = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])

def load_dataset(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

# Select 25 random images from each class
def get_random_images(dataset):
    random_images = {}
    for class_index, class_name in enumerate(dataset.classes):
        indices = np.where(np.array(dataset.targets) == class_index)[0]
        np.random.shuffle(indices)
        selected_indices = indices[:25]  # Select 25 random images
        random_images[class_name] = [dataset[idx] for idx in selected_indices]
    return random_images


def plot_class_distribution(dataset):
    classes = dataset.classes
    # Count the occurrence of each class
    _, counts = np.unique(dataset.targets, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='rebeccapurple')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{count}', ha='center', va='bottom')

    plt.show()


def display_sample_images(random_images):
    for class_name, images in random_images.items():
        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.1)
        fig.suptitle(f'Class: {class_name}', fontsize=16)

        for ax, (img, label) in zip(axes.flatten(), images):
            ax.axis('off')
            ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            ax.set_title(class_name, fontsize=8)
        
        plt.show()


def process_image_data(args):
    img, class_index = args
    img = np.transpose(img.numpy(), (1, 2, 0))
    histograms = []
    for i in range(3):  # RGB channels
        hist, bin_edges = np.histogram(img[:, :, i].ravel(), bins=256, range=(0, 1))
        histograms.append(hist)
    return class_index, histograms

def plot_pixel_intensity_distribution_parallel(random_images):
    image_data = [(img, class_index) for class_index, (_, images) in enumerate(random_images.items()) for img, _ in images]
    
    # Process images in parallel using a pool of processes
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_image_data, tqdm(image_data, desc="Processing Images"))

    # Reorganize results by class
    class_histograms = {class_name: [[] for _ in range(3)] for class_name in random_images.keys()}  # Prepare lists for RGB channels
    for class_index, histograms in results:
        class_name = list(random_images.keys())[class_index]
        for i in range(3):  # RGB channels
            class_histograms[class_name][i].append(histograms[i])

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    bin_edges = np.linspace(0, 1, 257)  # Assuming 256 bins from 0 to 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for class_index, (class_name, histograms) in enumerate(class_histograms.items()):
        for color_index, color in enumerate(['r', 'g', 'b']):
            # Aggregate histograms for the current color channel
            agg_hist = np.sum(histograms[color_index], axis=0)
            axes[class_index].bar(bin_centers, agg_hist, width=1/256, color=color, alpha=0.5, label=color.upper(), align='center')
        axes[class_index].set_title(class_name)
        axes[class_index].set_xlim([0, 1])
        axes[class_index].legend()

    plt.show()



if __name__ == '__main__':
    data_dir = './dataset'
    dataset = load_dataset(data_dir)
    random_images = get_random_images(dataset)
    plot_class_distribution(dataset)
    display_sample_images(random_images)
    plot_pixel_intensity_distribution_parallel(random_images)
    print('Done!')