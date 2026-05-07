import pandas as pd

df = pd.read_csv("../../Data/Raw Data/UCI_Credit_Card.csv")

df = df.rename(columns={"default.payment.next.month": "target"})

df = df.drop(columns=["ID"])

df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)

print("Dataset shape:", df.shape)
print()
print("Missing values:")
print(df.isnull().sum())
print()
print("Class distribution:")
print(df["target"].value_counts())

df.to_csv("cleaned_dataset3.csv", index=False)

print()
print("cleaned_dataset3.csv created")