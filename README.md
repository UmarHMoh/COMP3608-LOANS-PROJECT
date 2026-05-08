# COMP3608-LOANS-PROJECT

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas scikit-learn imbalanced-learn

1. Run Dataset 1 (Jupyter Notebook)
        cd Scripts/Dataset1
        jupyter notebook DS1-Cleaning_and_Modelling.ipynb

2. Run Dataset 2 (Loan Default)
        cd Scripts/Dataset2
        python cleanData.py
        python compare.py
        python tree.py
        python logistic.py
        python knn.py

3. Run Dataset 3 (UCI Credit Card)
        cd Scripts/Dataset3
        python cleanData.py
        python logistic.py
        python knn.py
        python tree.py
        python compare.py

What Each Script Does

Dataset 1
Script	Purpose
DS1-Cleaning_and_Modelling.ipynb	EDA and modeling in Jupyter
Project_Code.py	Python version of the notebook

Dataset 2 (Loan Default)
Script	Purpose
cleanData.py	Encodes categorical variables, drops LoanID
compare.py	Compares all 3 models
logistic.py	Logistic regression with SMOTE
knn.py	KNN with SMOTE + scaling
tree.py	Decision tree with SMOTE

Dataset 3 (UCI Credit Card)
Script	Purpose
cleanData.py	Cleans UCI dataset, renames target column, handles categorical values
logistic.py	Logistic regression with SMOTE + GridSearch
knn.py	KNN with SMOTE + scaling
tree.py	Decision tree with SMOTE
compare.py	Compares all 3 models

Model Features (Datasets 2 & 3)

SMOTE for handling class imbalance
Stratified K-Fold CV (5 folds)
GridSearchCV for hyperparameter tuning
Scoring: ROC-AUC (Dataset 2) / Recall (Dataset 3)

Dependencies
pandas
scikit-learn
imbalanced-learn
jupyter
openpyxl
