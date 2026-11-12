import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from predict import predict_news

st.title("📰 Fake News Detector")
text = st.text_area("Paste a news article here:", height=250)

if st.button("Check"):
    if text.strip():
        label, conf = predict_news(text)
        msg = f"This looks like **{label}** news"
        if conf is not None:
            msg += f" (confidence: {conf:.0%})"
        if label == "REAL":
            st.success(msg)
        else:
            st.error(msg)
    else:
        st.warning("Please enter some text.")