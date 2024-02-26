import os
import shutil
from PIL import Image
import cv2
from torchvision import transforms

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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


def detect_and_crop_face(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    # If no faces are detected, return None
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)  # Sort by area (w * h)
    x, y, w, h = faces[0]  # Choose the largest face
    # Crop the face from the image
    face_crop = img_cv[y:y + h, x:x + w]
    return face_crop


def process_image(img_path, processed_path, img_name):
    img_cv = cv2.imread(img_path)
    # Use the detect_and_crop_face function to get the cropped face image
    face_crop_cv = detect_and_crop_face(img_cv)
    # If a face was detected and cropped
    if face_crop_cv is not None:
        # Convert the cropped face to PIL Image
        face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop_cv, cv2.COLOR_BGR2RGB))

        img_transformed = transform_pipeline(face_crop_pil)
        img_processed = transforms.ToPILImage()(img_transformed)
        img_processed.save(os.path.join(processed_path, img_name))
    else:
        print(f"No face detected in {img_name}, skipped.")


expressions = ['Happy', 'Neutral', 'Suprised', 'Focused']

# Run the processing function for each expression
process_images('dataset', expressions)

print('Done!')
