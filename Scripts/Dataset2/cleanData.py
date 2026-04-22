import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/Data/Loan_prediction_default.csv")

print(df.shape)
print(df.isnull().sum().sum())

cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
            'HasDependents', 'LoanPurpose', 'HasCoSigner']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df = df.drop('LoanID', axis=1)

df.to_csv("data/Data/cleaned.csv", index=False)