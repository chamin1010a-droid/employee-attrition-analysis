"""데이터 구조 탐색 스크립트"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col} ({df[col].dtype}) - nunique: {df[col].nunique()}")

print(f"\n=== Target: Attrition ===")
print(df["Attrition"].value_counts())
attrition_rate = (df["Attrition"] == "Yes").mean() * 100
print(f"\nAttrition Rate: {attrition_rate:.1f}%")

print(f"\n=== Missing values ===")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values")
else:
    print(missing[missing > 0])

print(f"\n=== Constant columns (same value for all rows) ===")
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"  {col} = {df[col].iloc[0]}")

print(f"\n=== Sample rows (3 attrition Yes) ===")
print(df[df["Attrition"] == "Yes"].head(3).T)

print(f"\n=== Numeric columns summary ===")
print(df.describe().T[["mean", "std", "min", "max"]])
