import numpy as np
from src.data_preprocessing.preprocess_speech import extract_speech_features
from src.models.speech_emotion_model import build_speech_emotion_model
import os

# NOTE: This script assumes the existence of your real datasets.
def train_and_save():
    # 1. Load your dataset here (e.g., using pandas to read labels and audio paths)
    # 2. Extract features:
    # X = [extract_speech_features(path) for path in audio_paths]
    # y = [labels]
    
    # Dummy placeholder for shape verification
    X = np.random.rand(10, 100, 40)
    y = np.random.rand(10, 5) # 5 classes

    model = build_speech_emotion_model()
    model.fit(X, y, epochs=5, batch_size=2)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/speech_emotion_model.h5')
    print("Model trained and saved to models/speech_emotion_model.h5")

if __name__ == '__main__':
    train_and_save()
