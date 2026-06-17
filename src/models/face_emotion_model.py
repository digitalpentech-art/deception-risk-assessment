import random

class MockFaceModel:
    def predict(self, data):
        # Return dummy probabilities for 7 classes
        return [random.random() for _ in range(7)]

def build_face_emotion_model():
    return MockFaceModel()
