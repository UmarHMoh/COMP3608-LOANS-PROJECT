import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("lr", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "params": {
            "lr__C": [0.01, 0.1, 1, 10],
            "lr__solver": ["lbfgs"]
        }
    },
    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("knn", KNeighborsClassifier())
        ]),
        "params": {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        }
    },
    "Decision Tree": {
        "pipeline": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("tree", DecisionTreeClassifier(random_state=42))
        ]),
        "params": {
            "tree__max_depth": [3, 5, 7, 10, None],
            "tree__min_samples_split": [2, 5, 10],
            "tree__min_samples_leaf": [1, 2, 5],
            "tree__criterion": ["gini", "entropy"]
        }
    }
}

results = []

from sklearn.model_selection import cross_val_score

# Logistic Regression
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("lr", LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=2000,
        random_state=42
    ))
])

lr_scores = cross_val_score(
    lr_pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)

results.append({
    "Model": "Logistic Regression",
    "Best Parameters": {
        "lr__C": 0.01,
        "lr__solver": "lbfgs"
    },
    "Best CV Recall": round(lr_scores.mean(), 4),
    "Test Accuracy": round(accuracy_score(y_test, lr_pred), 4),
    "Test Precision": round(precision_score(y_test, lr_pred), 4),
    "Test Recall": round(recall_score(y_test, lr_pred), 4),
    "Test F1-score": round(f1_score(y_test, lr_pred), 4)
})

# KNN
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(
        n_neighbors=11,
        weights="uniform",
        metric="euclidean"
    ))
])

knn_scores = cross_val_score(
    knn_pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

knn_pipe.fit(X_train, y_train)
knn_pred = knn_pipe.predict(X_test)

results.append({
    "Model": "KNN",
    "Best Parameters": {
        "knn__metric": "euclidean",
        "knn__n_neighbors": 11,
        "knn__weights": "uniform"
    },
    "Best CV Recall": round(knn_scores.mean(), 4),
    "Test Accuracy": round(accuracy_score(y_test, knn_pred), 4),
    "Test Precision": round(precision_score(y_test, knn_pred), 4),
    "Test Recall": round(recall_score(y_test, knn_pred), 4),
    "Test F1-score": round(f1_score(y_test, knn_pred), 4)
})

# Decision Tree
tree_pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("tree", DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    ))
])

tree_scores = cross_val_score(
    tree_pipe,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

tree_pipe.fit(X_train, y_train)
tree_pred = tree_pipe.predict(X_test)

results.append({
    "Model": "Decision Tree",
    "Best Parameters": {
        "tree__criterion": "gini",
        "tree__max_depth": 3,
        "tree__min_samples_leaf": 1,
        "tree__min_samples_split": 2
    },
    "Best CV Recall": round(tree_scores.mean(), 4),
    "Test Accuracy": round(accuracy_score(y_test, tree_pred), 4),
    "Test Precision": round(precision_score(y_test, tree_pred), 4),
    "Test Recall": round(recall_score(y_test, tree_pred), 4),
    "Test F1-score": round(f1_score(y_test, tree_pred), 4)
})

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv("model_comparison_dataset3.csv", index=False)

print()
print("model_comparison_dataset3.csv created")