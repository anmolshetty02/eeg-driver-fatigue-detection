import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="EEG Drowsiness Detection", page_icon="🧠", layout="wide")

st.title("🧠 EEG-Based Driver Drowsiness Detection")
st.markdown("**Real-time fatigue detection using machine learning and brain signal analysis**")

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / 'models' / 'rf_model.pkl'
    scaler_path = Path(__file__).parent / 'models' / 'scaler.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ Model loaded successfully (98.7% accuracy)")
except Exception as e:
    st.error(f"❌ Model not found: {e}")
    st.stop()

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Demo: Upload EEG Data")
    
    uploaded_file = st.file_uploader("Upload CSV with EEG features", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} samples")
        st.dataframe(df.head())

with col2:
    st.subheader("🎯 Prediction Results")
    
    if uploaded_file and st.button("🔍 Analyze", type="primary"):
        with st.spinner("Analyzing EEG signals..."):
            X = df.drop('label', axis=1).values if 'label' in df.columns else df.values
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            
            avg_prob = probabilities.mean()
            
            st.metric("Drowsiness Score", f"{avg_prob*100:.1f}%")
            st.metric("Drowsy Samples", f"{predictions.sum()} / {len(predictions)}")
            
            if avg_prob > 0.7:
                st.error("⚠️ HIGH DROWSINESS - Driver needs break!")
            elif avg_prob > 0.4:
                st.warning("⚡ Moderate drowsiness detected")
            else:
                st.success("✅ Driver is alert")
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(probabilities)
            ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Drowsiness Probability')
            ax.set_title('Drowsiness Detection Over Time')
            ax.legend()
            st.pyplot(fig)

st.markdown("---")

with st.expander("📖 About This System"):
    st.markdown("""
    ### Performance Metrics
    - **Accuracy:** 98.7%
    - **AUC-ROC:** 1.000
    - **Precision:** 98% (Alert), 100% (Drowsy)
    - **Recall:** 100% (Alert), 97% (Drowsy)
    
    ### Features
    - Delta, Theta, Alpha, Beta power bands
    - Fatigue-specific ratios (θ/α, combined indices)
    - 10 engineered features total
    
    ### Model
    - Random Forest (100 trees, max_depth=10)
    - Validated on held-out test set
    
    ### Tech Stack
    - Python • scikit-learn • Signal Processing • Streamlit
    """)

st.markdown("---")
st.markdown("**Built by Anmol Shetty** | [GitHub](https://github.com/anmolshetty02)")
