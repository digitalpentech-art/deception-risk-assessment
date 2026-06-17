# Deployment Strategy

## Training (Google Colab)
- Use GPU-accelerated runtime.
- Save trained models (`.h5` files) to Google Drive.
- Export trained model weights for production.

## Inference (Flask Backend)
- Host Flask application on a secure server or cloud platform (e.g., AWS, GCP).
- Load pre-trained models on startup.
- Expose `/predict` API endpoint.

## Data Storage (Google Drive / Cloud Database)
- Models: Saved to Google Drive/Cloud storage.
- Session Data: Persisted in PostgreSQL database.
