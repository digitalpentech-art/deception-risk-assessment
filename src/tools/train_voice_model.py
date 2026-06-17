import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import os
from src.models.voice_stress_model import build_voice_stress_model

def train_and_save():
    # Placeholder for shape verification (15 features)
    X = np.random.rand(10, 15)
    y = np.random.rand(10, 3) # 3 stress classes

    model = build_voice_stress_model()
    model.fit(X, y, epochs=10, batch_size=2)
    
    os.makedirs('models', exist_ok=True)
    model.save('models/voice_stress_model.h5')
    print("Voice stress model trained and saved.")

if __name__ == '__main__':
    train_and_save()
