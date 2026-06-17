import random

class MockSpeechModel:
    def predict(self, data):
        # Return dummy probabilities for 5 classes
        return [random.random() for _ in range(5)]

def build_speech_emotion_model():
    return MockSpeechModel()
