import pandas as pd

df = pd.read_csv("data/Data/Loan_prediction_default.csv", encoding="latin1")

print("=" * 60)
print("DATA SUMMARY FOR Loan_prediction_default.csv")
print("=" * 60)

print(f"\n1. DATASET SHAPE:")
print(f"   {df.shape[0]} rows, {df.shape[1]} columns")

print(f"\n2. FIRST 5 ROWS:")
print(df.head())

print(f"\n3. COLUMN NAMES:")
for col in df.columns:
    print(f"   - {col}")

print(f"\n4. DATA TYPES:")
print(df.dtypes)

print(f"\n5. MISSING VALUES (%):")
missing = (df.isnull().sum() / len(df)) * 100
for col in df.columns:
    if missing[col] > 0:
        print(f"   {col}: {missing[col]:.2f}%")

print(f"\n6. STATUS COLUMN (Target):")
if 'Status' in df.columns:
    print(f"   Unique values: {df['Status'].unique()}")
    print(f"   Value counts:")
    print(df['Status'].value_counts())
else:
    print("   No 'Status' column found!")


summary = pd.DataFrame()
summary["Variable Name"] = df.columns
summary["Data Type"] = df.dtypes.values
summary["No. Unique Values"] = df.nunique().values
summary["Sample Values"] = summary["Variable Name"].map(
    lambda col: ", ".join(map(str, df[col].dropna().unique()[:5]))
)
summary["Missing %"] = summary["Variable Name"].map(df.isnull().mean() * 100)

summary.to_excel("data_summary_new.xlsx", index=False)
print(f"\n7. EXCEL SUMMARY SAVED:")
print("   File: data_summary_new.xlsx")
