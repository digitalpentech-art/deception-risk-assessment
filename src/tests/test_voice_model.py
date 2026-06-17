import unittest
from src.models.voice_stress_model import build_voice_stress_model
import numpy as np

class TestVoiceStressModel(unittest.TestCase):
    def test_model_build(self):
        model = build_voice_stress_model()
        self.assertIsNotNone(model)
        
    def test_model_output_shape(self):
        model = build_voice_stress_model()
        input_data = np.random.rand(1, 15)
        output = model.predict(input_data)
        self.assertEqual(output.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()
