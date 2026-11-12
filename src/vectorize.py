import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import CLEAN_DATA, VECTOR_DIR, MODELS_DIR, VECTORIZER_PATH

print("Loading cleaned data...")
df = pd.read_csv(CLEAN_DATA).dropna()
print("Rows:", len(df))

# 1. Split FIRST, before vectorizing (prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)

# 2. Fit TF-IDF on training data ONLY
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

print("Train matrix:", X_train_vec.shape)
print("Test matrix :", X_test_vec.shape)

# 3. Save everything
VECTOR_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
sparse.save_npz(VECTOR_DIR / "X_train.npz", X_train_vec)
sparse.save_npz(VECTOR_DIR / "X_test.npz", X_test_vec)
np.save(VECTOR_DIR / "y_train.npy", y_train.values)
np.save(VECTOR_DIR / "y_test.npy", y_test.values)
joblib.dump(vec, VECTORIZER_PATH)

print("Saved vectorized data in vectorized/ and vectorizer in models/")