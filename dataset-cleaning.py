import os
from PIL import Image
from torchvision import transforms

# Define the simplified transform pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def process_images(dataset_path):
    for expression in ['Happy', 'Neutral', 'Suprised']:
        expression_path = os.path.join(dataset_path, expression)
        processed_path = os.path.join(expression_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)

        for img_name in os.listdir(expression_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.img')):
                continue

            img_path = os.path.join(expression_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img_transformed = transform_pipeline(img)

            img_processed = transforms.ToPILImage()(img_transformed)
            img_processed.save(os.path.join(processed_path, img_name))


# Run the processing function
process_images('dataset')
print('Done!')
