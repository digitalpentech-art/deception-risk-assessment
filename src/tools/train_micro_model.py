import numpy as np
import os
from src.models.micro_model import build_micro_expression_model

def train_and_save():
    # Placeholder for shape verification
    X = np.random.rand(5, 16, 112, 112, 3)
    y = np.random.rand(5, 5) # 5 classes

    model = build_micro_expression_model()
    model.fit(X, y, epochs=5, batch_size=1)
    
    os.makedirs('models', exist_ok=True)
    model.save('models/micro_model.h5')
    print("Micro-expression model trained and saved.")

if __name__ == '__main__':
    train_and_save()
