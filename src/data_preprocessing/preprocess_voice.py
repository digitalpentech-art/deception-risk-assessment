import librosa
import numpy as np

def extract_voice_stress_features(audio_path):
    """
    Realistic extraction of voice stress features.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Fundamental frequency (pitch)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.mean(f0[voiced_flag]) if np.any(voiced_flag) else 0
    
    # Energy
    energy = np.sum(y**2)
    
    # MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    
    # Combine into a feature vector (1 + 1 + 13 = 15 features)
    features = np.hstack([pitch, energy, mfccs])
    return features
