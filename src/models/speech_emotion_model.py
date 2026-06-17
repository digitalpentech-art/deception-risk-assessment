import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_speech_emotion_model(num_classes=5):
    """
    Builds a Speech Emotion model using YAMNet pre-trained embeddings.
    """
    yamnet_model = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)
    
    model = Sequential([
        # YAMNet input is 1D audio waveform
        yamnet_model,
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
