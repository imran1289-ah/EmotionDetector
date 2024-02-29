import os
import shutil
from PIL import Image
from torchvision import transforms

# Define the simplified transform pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def delete_existing_processed_folder(expression_path):
    processed_path = os.path.join(expression_path, 'processed')
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)


def process_images(root_path, expressions):
    for expression in expressions:
        expression_path = os.path.join(root_path, expression)
        # Delete the existing 'processed' folder for the expression
        delete_existing_processed_folder(expression_path)
        # Create a new 'processed' directory
        processed_path = os.path.join(expression_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)

        if expression == 'Focused':
            for subfolder in os.listdir(expression_path):
                subfolder_path = os.path.join(expression_path, subfolder)
                if os.path.isdir(subfolder_path) and subfolder != 'processed':
                    process_subfolder(subfolder_path, processed_path)
        else:
            process_expression(expression_path, processed_path)


def process_subfolder(subfolder_path, processed_path):
    for img_name in os.listdir(subfolder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.img')):
            img_path = os.path.join(subfolder_path, img_name)
            process_image(img_path, processed_path, img_name)


def process_expression(expression_path, processed_path):
    for img_name in os.listdir(expression_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.img')):
            img_path = os.path.join(expression_path, img_name)
            process_image(img_path, processed_path, img_name)


def process_image(img_path, processed_path, img_name):
    img = Image.open(img_path).convert('RGB')
    img_transformed = transform_pipeline(img)
    img_processed = transforms.ToPILImage()(img_transformed)
    img_processed.save(os.path.join(processed_path, img_name))


expressions = ['Happy', 'Neutral', 'Suprised', 'Focused']

# Run the processing function for each expression
process_images('dataset', expressions)

print('Done!')
