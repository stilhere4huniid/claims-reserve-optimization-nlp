# Technical Documentation

## Tech Stack
- **Language:** Python 3.9+
- **NLP:** `sentence-transformers` (Model: `all-MiniLM-L6-v2`)
- **Model:** `XGBoost` (Gradient Boosting Regressor)
- **Explainability:** `SHAP` (Shapley Additive ExPlanations)
- **Interface:** `Streamlit`

## Data Pipeline
1. **Input:** Raw OSHA Severe Injury Narratives.
2. **Preprocessing:** Text cleaning and synthetic target engineering (Cost simulation).
3. **Feature Engineering:** Transformer-based embeddings (384 dimensions) generated from text narratives.
4. **Inference:** XGBoost model processes the 384-dimensional vector to output a continuous dollar value.

## Threshold Logic
The tool categorizes claims based on the predicted reserve:
- **Critical:** > $100,000
- **High:** > $60,000
- **Routine:** < $60,000