from flask import Flask, request, jsonify
from src.models.face_emotion_model import build_face_emotion_model
from src.models.speech_emotion_model import build_speech_emotion_model
from src.models.voice_stress_model import build_voice_stress_model
from src.models.micro_model import build_micro_expression_model
from src.models.risk_engine import assess_risk
import random

app = Flask(__name__)

# Load mock models
face_model = build_face_emotion_model()
speech_model = build_speech_emotion_model()
voice_model = build_voice_stress_model()
micro_model = build_micro_expression_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Simulate inference
    f_probs = face_model.predict([])
    s_probs = speech_model.predict([])
    v_probs = voice_model.predict([])
    m_probs = micro_model.predict([])
    
    # Simple mock fusion: combine probabilities to get 3 classes
    fused_probs = [random.random(), random.random(), random.random()]
    
    result = assess_risk(fused_probs)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
