import joblib
from utils import MODELS_DIR, VECTORIZER_PATH
from preprocessor import clean_text

model = joblib.load(MODELS_DIR / "best_model.pkl")
vec = joblib.load(VECTORIZER_PATH)

# Set this after checking the top words in evaluate.py
LABEL_1_MEANS = "REAL"   # change to "FAKE" if your check shows the reverse
LABEL_0_MEANS = "FAKE" if LABEL_1_MEANS == "REAL" else "REAL"

def predict_news(text: str):
    cleaned = clean_text(text)
    X = vec.transform([cleaned])
    pred = model.predict(X)[0]
    label = LABEL_1_MEANS if pred == 1 else LABEL_0_MEANS

    # Confidence works for any model that has either method
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())
    elif hasattr(model, "decision_function"):
        score = abs(float(model.decision_function(X)[0]))
        confidence = score / (1 + score)   # rough 0-1 scale, not a true probability
    else:
        confidence = None
    return label, confidence

if __name__ == "__main__":
    text = input("Paste a news article: ")
    label, conf = predict_news(text)
    print("Prediction:", label, "| Confidence:", None if conf is None else round(conf, 3))