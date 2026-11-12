import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from utils import VECTOR_DIR, MODELS_DIR

X_train = sparse.load_npz(VECTOR_DIR / "X_train.npz")
X_test = sparse.load_npz(VECTOR_DIR / "X_test.npz")
y_train = np.load(VECTOR_DIR / "y_train.npy")
y_test = np.load(VECTOR_DIR / "y_test.npy")

# Shortlist from LazyPredict (edit to match your leaderboard)
models = {
    "LinearSVC": LinearSVC(),
    "Ridge": RidgeClassifier(),
    "PassiveAggressive": PassiveAggressiveClassifier(random_state=42),
    "SGD": SGDClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

results = []
best_name, best_f1, best_model = None, 0, None

for name, model in models.items():
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {seconds:.1f}s")
    results.append({"model": name, "accuracy": acc, "f1": f1, "train_seconds": seconds})

    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, model

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(
    MODELS_DIR / "final_results.csv", index=False
)
print(f"\nBest model: {best_name} (F1 = {best_f1:.4f}) saved to models/best_model.pkl")