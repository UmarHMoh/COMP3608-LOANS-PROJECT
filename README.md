# COMP3608-LOANS-PROJECT

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

requirements.txt includes the following:
```bash
pandas
scikit-learn
imbalanced-learn
jupyter
openpyxl
```


## Step 2: Run the Datasets

1. Run Dataset 1 (Jupyter Notebook)
```bash
cd Scripts/Dataset1
jupyter notebook DS1-Cleaning_and_Modelling.ipynb
```

2. Run Dataset 2 (Loan Default)
```bash
cd Scripts/Dataset2
python cleanData.py
python compare.py
python tree.py
python logistic.py
python knn.py
```

3. Run Dataset 3 (UCI Credit Card)
```bash
cd Scripts/Dataset3
python cleanData.py
python logistic.py
python knn.py
python tree.py
python compare.py
```


### What Each Script Does

| Dataset | Script | Purpose |
|--------|--------|---------|
| Dataset 1 | DS1-Cleaning_and_Modelling.ipynb | EDA and modelling in Jupyter |
| Dataset 1 | Project_Code.py | Python version of the notebook |
| Dataset 2 (Loan Default) | cleanData.py | Encodes categorical variables, drops LoanID |
| Dataset 2 (Loan Default) | compare.py | Compares all 3 models |
| Dataset 2 (Loan Default) | logistic.py | Logistic regression with SMOTE |
| Dataset 2 (Loan Default) | knn.py | KNN with SMOTE + scaling |
| Dataset 2 (Loan Default) | tree.py | Decision tree with SMOTE |
| Dataset 3 (UCI Credit Card) | cleanData.py | Cleans dataset, renames target column |
| Dataset 3 (UCI Credit Card) | logistic.py | Logistic regression with SMOTE + GridSearch |
| Dataset 3 (UCI Credit Card) | knn.py | KNN with SMOTE + scaling |
| Dataset 3 (UCI Credit Card) | tree.py | Decision tree with SMOTE |
| Dataset 3 (UCI Credit Card) | compare.py | Compares all 3 models |


### Model Features (Datasets 2 & 3)

- SMOTE for handling class imbalance  
- Stratified K-Fold CV (5 folds)  
- GridSearchCV for hyperparameter tuning  
- Scoring:
  - ROC-AUC (Dataset 2)
  - Recall (Dataset 3)
