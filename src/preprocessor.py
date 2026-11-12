import pandas as pd
from utils import RAW_DATA, CLEAN_DATA, clean_text

def main():
    print("Loading data...")
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA}")

    df = pd.read_csv(RAW_DATA)
    print("Raw rows:", len(df))

    df = df.dropna(subset=["text", "label"]).drop_duplicates()
    df["title"] = df["title"].fillna("")

    print("Cleaning text (this takes a few minutes)...")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    # Remove empty rows and duplicates created by cleaning
    df = df[df["content"].str.strip() != ""]
    df = df.drop_duplicates(subset="content")

    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df[["content", "label"]].to_csv(CLEAN_DATA, index=False)

    print(f"Saved {len(df)} rows to {CLEAN_DATA}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()