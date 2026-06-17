import cv2
import numpy as np

def extract_micro_expression_frames(video_path, num_frames=16, target_size=(112, 112)):
    """
    Realistic extraction: Extracts a sequence of frames from a video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frames.append(frame / 255.0)
    
    cap.release()
    return np.array(frames)
