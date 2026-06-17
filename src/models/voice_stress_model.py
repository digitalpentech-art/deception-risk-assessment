import random

class MockVoiceModel:
    def predict(self, data):
        # Return dummy probabilities for 3 stress classes
        return [random.random() for _ in range(3)]

def build_voice_stress_model():
    return MockVoiceModel()
