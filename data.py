import os
import pandas as pd

def load_split(path):
    rows = []
    for sentiment in ["pos", "neg"]:
        folder = os.path.join(path, sentiment)
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                fpath = os.path.join(folder, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                rows.append({
                    "review": text,
                    "sentiment": 1 if sentiment == "pos" else 0
                })
    return pd.DataFrame(rows)

# Load directly from your Jupyter home folders
train_df = load_split("train")
test_df = load_split("test")

# Save as CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Preview a few rows
print(train_df.shape)   # (25000, 2)
print(train_df.head())
print("\n")
print(train_df.tail())

print(test_df.shape)    # (25000, 2)
print(test_df.head())
print("\n")
print(test_df.tail())
