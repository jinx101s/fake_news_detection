import streamlit as st
from predict import predict_news

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("TF-IDF + linear classifier trained on the WELFake dataset.")

text = st.text_area("Paste a news article here:", height=250)

if st.button("Check"):
    if len(text.split()) < 20:
        st.warning("Please paste a longer article (at least ~20 words).")
    else:
        label, conf = predict_news(text)
        msg = f"This looks like **{label}** news"
        if conf is not None:
            msg += f" (confidence: {conf:.0%})"
        (st.success if label == "REAL" else st.error)(msg)

st.info(
    "Prototype: the model learns writing style and source patterns from its "
    "training data. It does not fact-check claims, so treat results as a hint."
)