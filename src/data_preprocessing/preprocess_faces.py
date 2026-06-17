import cv2
import numpy as np
import os

def preprocess_face(image_path, target_size=(224, 224)):
    """
    Resizes, normalizes, and prepares face images for the model.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

# TODO: Add logic for augmentation and splitting train/test
