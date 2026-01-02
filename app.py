import streamlit as st
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import joblib
import json
import matplotlib.pyplot as plt
import shap

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Claims Reserve Optimizer",
    page_icon="🛡️",
    layout="wide"
)

# --- LOAD ASSETS (Cached for performance) ---
@st.cache_resource
def load_resources():
    # Load the model into a Scikit-Learn wrapper to maintain compatibility with SHAP
    model = xgb.XGBRegressor()
    model.load_model("models/claims_model.json")
    
    # Load the NLP Transformer
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load SHAP assets
    explainer = joblib.load("models/shap_explainer.pkl")
    background_data = pd.read_pickle("models/shap_background.pkl")
    
    return model, nlp_model, explainer, background_data

# Initialize the assets
try:
    model, nlp, explainer, background = load_resources()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- SIDEBAR / INPUT ---
st.sidebar.header("Claim Details")
st.sidebar.info("Enter the First Notice of Loss (FNOL) narrative below to calculate the optimized reserve.")

claim_input = st.sidebar.text_area(
    "Accident Description:",
    "Employee was struck by a falling pallet in the warehouse, resulting in a fractured leg and emergency hospitalization.",
    height=200
)

# --- MAIN INTERFACE ---
st.title("🛡️ Insurance Claims Reserve Optimization Tool")
st.markdown("""
This tool uses **Natural Language Processing (NLP)** and **Gradient Boosting** to estimate 
claim severity based on unstructured accident reports. This helps insurers set accurate 
reserves immediately upon filing.
""")
st.markdown("---")

if st.sidebar.button("Run Optimization Analysis"):
    with st.spinner("Analyzing narrative and calculating risk drivers..."):
        # 1. Transform text to embedding
        embedding = nlp.encode([claim_input])
        df_input = pd.DataFrame(embedding, columns=[f'nlp_{i}' for i in range(384)])
        
        # 2. Predict
        prediction = model.predict(df_input)[0]
        
        # --- RESULTS LAYOUT ---
        st.success("Analysis Complete!")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Financial Estimate")
            st.metric(label="Recommended Reserve", value=f"${prediction:,.2f}")
            
            # --- UPDATED LOGIC FOR SEVERITY BADGES ---
            if prediction > 80000:
                st.error("Priority: CRITICAL")
                st.write("**Assessment:** Catastrophic event detected (likely amputation or multi-system trauma).")
            elif prediction > 60000:
                st.warning("Priority: HIGH")
                st.write("**Assessment:** Serious injury involving prolonged hospitalization.")
            else:
                st.info("Priority: ROUTINE")
                st.write("**Assessment:** Standard severe injury within baseline OSHA parameters.")

        with col2:
            st.subheader("Explainability (SHAP)")
            # Calculate local SHAP values for this input
            shap_vals = explainer.shap_values(df_input)
            
            # Generate the Force Plot
            fig = shap.force_plot(
                explainer.expected_value, 
                shap_vals[0, :], 
                df_input.iloc[0, :], 
                matplotlib=True, 
                show=False
            )
            st.pyplot(plt.gcf())
            plt.close()

# --- FOOTER ---
st.markdown("---")
st.caption("Developed as a Data Science Portfolio Project | Tech Stack: Python, XGBoost, Transformers, SHAP, Streamlit")