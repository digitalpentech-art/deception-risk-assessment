# Dataset Selection

## Overview
The following datasets are required for training and evaluating the multimodal AI system.

| Modality | Dataset | Purpose |
| :--- | :--- | :--- |
| **Facial Emotion** | [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Training CNN for emotion classification. |
| **Speech Emotion** | [RAVDESS](https://zenodo.org/record/1188976) | Training Speech Emotion Model. |
| **Micro-expression** | [CASME II](https://casme.psych.ac.cn/casme/index) | Training Micro-expression detection. |
| **Voice Stress** | (Custom/Derived) | Extracted features (Pitch, Energy, Jitter, Shimmer, MFCC). |

## Implementation Notes
- Data needs to be downloaded, preprocessed (resize, normalize, augmentation), and stored in `data/raw/` and `data/processed/` directories.
- Split ratios (Train/Test) to be defined during implementation.
