import random

class MockMicroModel:
    def predict(self, data):
        # Return dummy probabilities for 5 classes
        return [random.random() for _ in range(5)]

def build_micro_expression_model():
    return MockMicroModel()
