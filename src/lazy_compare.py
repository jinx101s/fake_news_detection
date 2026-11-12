import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from lazypredict.Supervised import LazyClassifier
from utils import VECTOR_DIR, MODELS_DIR

X_train = sparse.load_npz(VECTOR_DIR / "X_train.npz")
X_test = sparse.load_npz(VECTOR_DIR / "X_test.npz")
y_train = np.load(VECTOR_DIR / "y_train.npy")
y_test = np.load(VECTOR_DIR / "y_test.npy")

# Sample rows so it finishes in reasonable time
rng = np.random.RandomState(42)
tr_idx = rng.choice(X_train.shape[0], size=10000, replace=False)
te_idx = rng.choice(X_test.shape[0], size=3000, replace=False)
X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
X_te, y_te = X_test[te_idx], y_test[te_idx]

# Reduce 50,000 sparse features to 100 dense features
svd = TruncatedSVD(n_components=100, random_state=42)
X_tr_d = svd.fit_transform(X_tr)
X_te_d = svd.transform(X_te)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, _ = clf.fit(X_tr_d, X_te_d, y_tr, y_te)

print(models.head(15))
MODELS_DIR.mkdir(exist_ok=True)
models.to_csv(MODELS_DIR / "lazypredict_results.csv")
print("\nSaved to models/lazypredict_results.csv")