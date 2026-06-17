import numpy as np
import os
from src.data_preprocessing.preprocess_faces import preprocess_face
from src.models.face_emotion_model import build_face_emotion_model

def train_and_save():
    # 1. Load your dataset (e.g., list of image paths and labels)
    # 2. X = np.array([preprocess_face(path) for path in paths])
    
    # Placeholder for shape verification
    X = np.random.rand(10, 224, 224, 3)
    y = np.random.rand(10, 7) # 7 classes for FER2013

    model = build_face_emotion_model()
    model.fit(X, y, epochs=5, batch_size=2)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/face_emotion_model.h5')
    print("Face model trained and saved.")

if __name__ == '__main__':
    train_and_save()
