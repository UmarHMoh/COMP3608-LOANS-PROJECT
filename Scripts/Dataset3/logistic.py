import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


df = pd.read_csv("cleaned_dataset3.csv")

y = df["target"]
X = df.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("lr", LogisticRegression(max_iter=2000, random_state=42))
])

params = {
    "lr__C": [0.01, 0.1, 1, 10],
    "lr__solver": ["lbfgs"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    params,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print("Logistic Regression Results")
print("Best Params:", grid.best_params_)
print("Best CV Recall:", grid.best_score_)
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))