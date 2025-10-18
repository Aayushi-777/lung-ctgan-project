import pandas as pd

for file in ["data/breast.csv", "data/heart.csv", "data/lung.csv"]:
    df = pd.read_csv(file)
    print(f"\n{file} columns:")
    print(df.columns.tolist())
