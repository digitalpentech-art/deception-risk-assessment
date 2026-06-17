# Final Project Report

## Chapter 1: Introduction
This project presents a multimodal deep learning system for behavioral risk assessment during suspect interrogation. By integrating facial emotion recognition, speech emotion, voice stress analysis, and micro-expression detection, the system provides an objective, assistive behavioral indicator to aid investigators. The system explicitly adheres to ethical guidelines, functioning as an assistive research tool rather than a deterministic lie detector.

## Chapter 2: Literature Review (Selected References)
*   **Facial Analysis:** Goodfellow et al. (2013) on the FER2013 dataset; current state-of-the-art using ResNet-based transfer learning.
*   **Speech Emotion:** Livingstone & Russo (2018) on the RAVDESS dataset.
*   **Micro-expressions:** Yan et al. (2014) on the CASME II dataset and the application of 3D CNNs for temporal feature extraction.
*   **Multimodal Fusion:** Recent advancements in Transformer-based cross-attention mechanisms for synchronizing heterogeneous behavioral modalities.

## Chapter 3: System Design
The system utilizes a modular, microservice-based architecture:
1.  **Preprocessing Layer:** Standardizes video/audio input.
2.  **Model Layer:** Employs transfer learning (ResNet50, YAMNet, I3D) for feature extraction.
3.  **Fusion Layer:** Transformer-based architecture for cross-modal interaction.
4.  **Backend/Frontend:** Flask-based API serving a Bootstrap-styled dashboard.

## Chapter 4: Implementation
The system is built on Python and TensorFlow. Key implementations include:
*   **Preprocessing:** `src/data_preprocessing/` (OpenCV/Librosa)
*   **Models:** `src/models/` (TensorFlow Hub/Keras)
*   **Backend API:** `src/backend/app.py`
*   **Containerization:** `Dockerfile` and `render.yaml` for production-ready deployment.

## Chapter 5: Summary & Recommendations
The system provides a robust framework for behavioral analysis. **Recommendations:** Future work should focus on large-scale dataset acquisition, rigorous validation against diverse demographic groups, and deployment of edge-computing hardware to minimize inference latency for real-time applications.
