import librosa
import numpy as np

def extract_speech_features(audio_path):
    """
    Extracts MFCC and Mel Spectrogram from audio files.
    """
    y, sr = librosa.load(audio_path)
    # Noise reduction and feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return mfcc, mel_spec

# TODO: Implement Pitch extraction and stress analysis features
