from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "WELFake_Dataset.csv"
CLEAN_DATA = BASE_DIR / "dataset" / "cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
MODELS_DIR = BASE_DIR / "models"
VECTOR_DIR = BASE_DIR / "vectorized"

LABEL_MAP = {0: "Fake", 1: "Real"}


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)