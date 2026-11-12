import pandas as pd
from utils import RAW_DATA

df = pd.read_csv(RAW_DATA)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nLabel distribution:\n", df["label"].value_counts())


for label in [0, 1]:
    print(f"\n===== 3 sample titles with label = {label} =====")
    for t in df[df["label"] == label]["title"].dropna().sample(3, random_state=1):
        print("-", t)