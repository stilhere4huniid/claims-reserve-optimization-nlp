# Methodology: Claims Reserve Optimization

## 1. Data Selection
We utilized the **OSHA Severe Injury Report (2015-2025)**. This dataset was selected because it contains high-quality, human-written narratives of industrial accidents, providing a realistic testbed for NLP.

## 2. Feature Extraction (NLP)
Rather than using simple keyword matching (TF-IDF), we used **Sentence Transformers**. This allows the model to understand context—for example, recognizing that a "fractured limb" is semantically closer to "surgical intervention" than a "minor scratch."

## 3. Modeling
We selected **XGBoost** for its ability to handle high-dimensional sparse data.
- **Samples:** 50,000 records.
- **Split:** 80% Train / 20% Test.
- **Metrics:** We prioritized **Mean Absolute Error (MAE)** to ensure the dollar "miss" was minimized for financial reserving.

## 4. Explainability (SHAP)
To overcome the "Black Box" nature of Gradient Boosting, we integrated **SHAP**. This breaks down the model's logic into "positive" and "negative" forces, proving to stakeholders that the model is correctly identifying high-risk keywords like "amputation" and "hospitalization."

## 🚀 Future Roadmap & Model Evolution
To move this from a Proof-of-Concept (PoC) to a production-grade insurance tool, the following iterations are proposed:
1. **Hybrid Data Integration:** Incorporate structured features (Claimant Age, Tenured Status, State-specific Medical Indices) to improve precision.
2. **Hyperparameter Tuning:** Implement `Optuna` for Bayesian optimization of the XGBoost learning rate and tree depth.
3. **Low-Signal Fallback:** Develop a secondary "Heuristic Model" to handle "Thin File" narratives where NLP embeddings lack sufficient variance.
