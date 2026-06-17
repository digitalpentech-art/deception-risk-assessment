import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

def build_micro_expression_model(num_classes=5):
    """
    Builds a Micro-expression model using I3D Kinetics-400 pre-trained model.
    """
    i3d_model = hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-400/1", trainable=False)
    
    model = Sequential([
        i3d_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
