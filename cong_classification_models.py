import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

df_congestion = pd.read_csv("agg_cong_packets.csv")

df_no_congestion = pd.read_csv("agg_non_cong_packets.csv")

# combine both into one dataset
df = pd.concat([df_congestion, df_no_congestion], ignore_index=True)

# drop raw time so the model does not just learn capture timing
df = df.drop(columns=["time_window"], errors="ignore")

# target column
label_col = "label"

# X = input features, y = labels
X = df.drop(columns=[label_col])
y = df[label_col]

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# define models to compare
models = {
    "Logistic Regression": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # scale features because logistic regression works better when features are on similar scales
        ("scaler", StandardScaler()),

        # simple baseline linear classifier
        ("model", LogisticRegression(
            max_iter=1000,   # allows more iterations so training converges
            C=1.0,           # inverse regularization strength
            solver="lbfgs",
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # single tree model that is easy to interpret
        ("model", DecisionTreeClassifier(
            max_depth=8,          # limits how deep the tree can grow
            min_samples_split=5,  # minimum samples needed before splitting
            min_samples_leaf=2,   # minimum samples allowed in a leaf
            criterion="gini",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # ensemble of decision trees, usually strong for tabular data
        ("model", RandomForestClassifier(
            n_estimators=200,      # number of trees
            max_depth=10,          # limits tree depth
            min_samples_split=5,   # minimum samples needed before splitting
            min_samples_leaf=2,    # minimum samples allowed in a leaf
            max_features="sqrt",   # number of features considered per split
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # similar to random forest but uses more random split choices
        ("model", ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "KNN": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # scale features because KNN is based on distance
        ("scaler", StandardScaler()),

        # nearest-neighbor classifier
        ("model", KNeighborsClassifier(
            n_neighbors=5,      # number of neighbors used for voting
            weights="distance", # closer neighbors matter more
            metric="minkowski",
            p=2                 # p=2 is Euclidean distance
        ))
    ]),

    "SVM": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # scale features because SVM is very sensitive to scale
        ("scaler", StandardScaler()),

        # nonlinear classifier using an RBF kernel
        ("model", SVC(
            kernel="rbf",
            C=1.0,          # regularization strength
            gamma="scale",  # controls how local the decision boundary is
            probability=True,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        # fill missing numeric values
        ("imputer", SimpleImputer(strategy="median")),

        # boosted tree model, often one of the best for structured data
        ("model", XGBClassifier(
            n_estimators=200,        # number of boosting trees
            max_depth=6,             # tree depth
            learning_rate=0.1,       # step size for each boosting round
            subsample=0.8,           # fraction of rows used per tree
            colsample_bytree=0.8,    # fraction of features used per tree
            min_child_weight=1,      # minimum weight needed in a child node
            reg_lambda=1.0,          # L2 regularization
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

# store results for comparison
results = []

# train and evaluate each model
for name, pipeline in models.items():
    print(f"\n--- {name} ---")

    # fit model on training data
    pipeline.fit(X_train, y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)

    # calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # save metrics for later comparison
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    # print detailed results
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Congestion", "Congestion"],
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# compare all models by F1
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)

print("\n=== Model Comparison ===")
print(results_df)

# save comparison results
results_df.to_csv("model_comparison_congestion.csv", index=False)
print("\nSaved model comparison to model_comparison_congestion.csv")

# output:
# --- Logistic Regression ---
# Accuracy: 0.689119170984456
# Precision: 0.7323943661971831
# Recall: 0.5591397849462365
# F1: 0.6341463414634146

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.66      0.81      0.73       100
#    Congestion       0.73      0.56      0.63        93

#      accuracy                           0.69       193
#     macro avg       0.70      0.68      0.68       193
#  weighted avg       0.70      0.69      0.68       193

# Confusion Matrix:
# [[81 19]
#  [41 52]]

# --- Decision Tree ---
# Accuracy: 0.7823834196891192
# Precision: 0.7684210526315789
# Recall: 0.7849462365591398
# F1: 0.776595744680851

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.80      0.78      0.79       100
#    Congestion       0.77      0.78      0.78        93

#      accuracy                           0.78       193
#     macro avg       0.78      0.78      0.78       193
#  weighted avg       0.78      0.78      0.78       193

# Confusion Matrix:
# [[78 22]
#  [20 73]]

# --- Random Forest ---
# Accuracy: 0.8082901554404145
# Precision: 0.8181818181818182
# Recall: 0.7741935483870968
# F1: 0.7955801104972375

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.80      0.84      0.82       100
#    Congestion       0.82      0.77      0.80        93

#      accuracy                           0.81       193
#     macro avg       0.81      0.81      0.81       193
#  weighted avg       0.81      0.81      0.81       193

# Confusion Matrix:
# [[84 16]
#  [21 72]]

# --- Extra Trees ---
# Accuracy: 0.7564766839378239
# Precision: 0.7875
# Recall: 0.6774193548387096
# F1: 0.7283236994219653

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.73      0.83      0.78       100
#    Congestion       0.79      0.68      0.73        93

#      accuracy                           0.76       193
#     macro avg       0.76      0.75      0.75       193
#  weighted avg       0.76      0.76      0.75       193

# Confusion Matrix:
# [[83 17]
#  [30 63]]

# --- KNN ---
# Accuracy: 0.7409326424870466
# Precision: 0.7047619047619048
# Recall: 0.7956989247311828
# F1: 0.7474747474747475

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.78      0.69      0.73       100
#    Congestion       0.70      0.80      0.75        93

#      accuracy                           0.74       193
#     macro avg       0.74      0.74      0.74       193
#  weighted avg       0.75      0.74      0.74       193

# Confusion Matrix:
# [[69 31]
#  [19 74]]

# --- SVM ---
# Accuracy: 0.6787564766839378
# Precision: 0.7246376811594203
# Recall: 0.5376344086021505
# F1: 0.6172839506172839

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.65      0.81      0.72       100
#    Congestion       0.72      0.54      0.62        93

#      accuracy                           0.68       193
#     macro avg       0.69      0.67      0.67       193
#  weighted avg       0.69      0.68      0.67       193

# Confusion Matrix:
# [[81 19]
#  [43 50]]

# --- XGBoost ---
# Accuracy: 0.7979274611398963
# Precision: 0.8068181818181818
# Recall: 0.7634408602150538
# F1: 0.7845303867403315

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.79      0.83      0.81       100
#    Congestion       0.81      0.76      0.78        93

#      accuracy                           0.80       193
#     macro avg       0.80      0.80      0.80       193
#  weighted avg       0.80      0.80      0.80       193

# Confusion Matrix:
# [[83 17]
#  [22 71]]

# === Model Comparison ===
#                  Model  Accuracy  Precision    Recall        F1
# 2        Random Forest  0.808290   0.818182  0.774194  0.795580
# 6              XGBoost  0.797927   0.806818  0.763441  0.784530
# 1        Decision Tree  0.782383   0.768421  0.784946  0.776596
# 4                  KNN  0.740933   0.704762  0.795699  0.747475
# 3          Extra Trees  0.756477   0.787500  0.677419  0.728324
# 0  Logistic Regression  0.689119   0.732394  0.559140  0.634146
# 5                  SVM  0.678756   0.724638  0.537634  0.617284

