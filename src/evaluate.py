import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from utils import VECTOR_DIR, MODELS_DIR, VECTORIZER_PATH

X_train = sparse.load_npz(VECTOR_DIR / "X_train.npz")
X_test = sparse.load_npz(VECTOR_DIR / "X_test.npz")
y_train = np.load(VECTOR_DIR / "y_train.npy")
y_test = np.load(VECTOR_DIR / "y_test.npy")

model = joblib.load(MODELS_DIR / "best_model.pkl")
vec = joblib.load(VECTORIZER_PATH)
print("Evaluating:", type(model).__name__)

# 1. Report on the held-out test set
preds = model.predict(X_test)
print(classification_report(y_test, preds, target_names=["Label 0", "Label 1"]))

# 2. Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(MODELS_DIR / "confusion_matrix.png")
plt.show()

# 3. Cross-validation (uses a fresh copy so best_model.pkl is untouched)
scores = cross_val_score(clone(model), X_train, y_train, cv=5, scoring="f1")
print("CV F1 scores:", np.round(scores, 4))
print(f"Mean: {scores.mean():.4f} | Std: {scores.std():.4f}")

# 4. Top words (linear models only)
if hasattr(model, "coef_"):
    names = np.array(vec.get_feature_names_out())
    coefs = model.coef_[0]
    print("\nTop words for label 1:", names[np.argsort(coefs)[-15:]])
    print("Top words for label 0:", names[np.argsort(coefs)[:15]])