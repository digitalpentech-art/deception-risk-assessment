import librosa
import numpy as np

def extract_speech_features(audio_path, n_mfcc=40, max_len=100):
    """
    Extracts MFCC features from an audio file.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
            
        return mfcc.T # Return (time_steps, n_mfcc)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
