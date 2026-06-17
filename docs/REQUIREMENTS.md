# Requirements Analysis

## Functional Requirements
- Upload video (for facial/micro-expression analysis).
- Upload audio (for speech emotion/voice stress analysis).
- Detect facial emotions (FER2013).
- Analyze speech emotions (RAVDESS).
- Analyze voice stress (Pitch, Energy, Jitter, Shimmer, MFCC).
- Detect micro-expressions (CASME II).
- Perform multimodal fusion (Transformer-based).
- Display behavioral risk score (Low, Medium, High).
- Generate reports.

## Non-functional Requirements
- Accuracy > 85%
- Modular architecture.
- Scalability.
- Security.
- Explainability (SHAP, Attention Visualization).
- Low latency.

## Ethical Constraints
- MUST NOT: Predict truth or lie, determine guilt or innocence.
- MUST clearly state assistive nature and limitation in legal settings.
