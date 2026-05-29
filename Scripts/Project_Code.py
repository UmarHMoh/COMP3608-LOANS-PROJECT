import pandas as pd


url = "https://raw.githubusercontent.com/UmarHMoh/COMP3608-LOANS-PROJECT/refs/heads/datasets-Load_Default.csv/data/Loan_Default.csv"

#df=pd.read_csv("Loan_Default.csv")
df = pd.read_csv(url, encoding="latin1", sep=",")

#print(df.shape)
#print(df.head())
#print(df.info())

summary=pd.DataFrame()
# All variables
summary["Variable Name"]=df.columns
summary["Data Type"]=df.dtypes.values
summary["No. Unique Values"] = df.nunique().values
summary["Sample Values"] = summary["Variable Name"].map(
    lambda col: ", ".join(map(str, df[col].dropna().unique()[:5]))
)
summary["Missing %"] = summary["Variable Name"].map(df.isnull().mean() * 100)
# Categorical columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns
summary["Top Category Frequency (%)"] = None
for col in cat_cols:
    top_freq = df[col].value_counts(normalize=True).iloc[0] * 100
    summary.loc[summary["Variable Name"] == col, "Top Category Frequency (%)"] = top_freq
#Numerical Columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
summary["Num: Std"] = summary["Variable Name"].map(df.std(numeric_only=True))
summary["Num: Min"] = summary["Variable Name"].map(df.min(numeric_only=True))
summary["Num: Max"] = summary["Variable Name"].map(df.max(numeric_only=True))
correlations = df.corr(numeric_only=True)["Status"]
summary["Correlation with Status"] = summary["Variable Name"].map(correlations)
#print(summary)
summary.to_excel("data_summary.xlsx", index=False)




