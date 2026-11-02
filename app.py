from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_model_pipeline.joblib"

st.set_page_config(page_title="Customer Review Sentiment", page_icon="💬", layout="centered")
st.title("Customer Review Sentiment Analysis")
st.caption("AAI-003 | TF-IDF + Machine Learning")

@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run the notebook to generate it first."
        )
    return joblib.load(MODEL_PATH)

try:
    bundle = load_pipeline()
    model = bundle["model"]
    vectorizer = bundle["tfidf"]
    model_name = bundle["model_name"]
    st.success(f"Loaded model: {model_name}")
    review = st.text_area("Enter a customer review", placeholder="Write your review here...")
    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            features = vectorizer.transform([review.strip()])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities) * 100)
            st.subheader(f"Predicted sentiment: {prediction}")
            st.metric("Estimated confidence", f"{confidence:.1f}%")
            st.caption("Confidence is a model estimate, not a guarantee. Ask a human to review ambiguous feedback.")
except FileNotFoundError as exc:
    st.error(str(exc))
