import pandas as pd
import numpy as np

path = "Data/UNSW_NB15_training-set.csv" 

try:
    df_head = pd.read_csv(path, sep=None, engine="python", nrows=5)
    print("Detected columns (first 5 rows read):")
    print(df_head.head())
except Exception as e:
    print("Auto-detect failed:", e)
    df_head = pd.read_csv(path, header=None, nrows=5)
    print("Read with header=None. Sample rows:")
    print(df_head)

chunksize = 200000
reader = pd.read_csv(path, sep=None, engine="python", chunksize=chunksize)
sample = next(reader)
print("Sample shape:", sample.shape)

if any(col.lower() in ["label","attack_cat","attack-cat","attack category"] for col in sample.columns):
    print("Column names found:", sample.columns.tolist())
else:
    print("No obvious header names present. Inspecting last two columns by position:")
    print(sample.iloc[:, -5:-0].head())  

df = sample.copy()
print("Dtype overview:")
print(df.dtypes.value_counts())
print("\nMissing values count (sample chunk):")
print(df.isna().sum().sort_values(ascending=False).head(20))


if "label" in df.columns:
    print("Label distribution (sample):")
    print(df['label'].value_counts(normalize=True))
else:
    y = df.iloc[:, -1]
    print("Inferred label (last column) unique values:", y.unique(), "counts:")
    print(y.value_counts(normalize=True))
