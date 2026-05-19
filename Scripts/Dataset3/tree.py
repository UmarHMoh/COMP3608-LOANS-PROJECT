import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt


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

from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("tree", DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("Decision Tree Results")
print("Best Params:", {
    "tree__criterion": "gini",
    "tree__max_depth": 3,
    "tree__min_samples_leaf": 1,
    "tree__min_samples_split": 2
})
print("Best CV Recall:", cv_scores.mean())
print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

y_probs = pipe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {auc_score:.3f})")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree Dataset 3")
plt.legend()

plt.show()
