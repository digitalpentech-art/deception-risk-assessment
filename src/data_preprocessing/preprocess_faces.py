import cv2
import numpy as np

def preprocess_face(image_path, target_size=(224, 224)):
    """
    Realistic preprocessing: loads, resizes, and normalizes images using OpenCV.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img
