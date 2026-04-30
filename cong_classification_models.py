import pandas as pd
import joblib

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
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

# load aggregated dataset
df = pd.read_csv("aggregated_dataset_5.csv")

# columns not useful for congestion prediction
drop_cols = [
    "source_file",
    "time_window",
    "window_index",
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

# keep a full version for later analysis
df_full = df.copy()

# target column
label_col = "congestion_label"

# full version (with everything)
X_full = df_full.drop(columns=[label_col])
y_full = df_full[label_col]

# model version (drop leakage + traffic labels)
X_model = X_full.drop(columns=drop_cols, errors="ignore")

print("Features used for training:")
print(X_model.columns.tolist())

# split both versions together so rows stay aligned
X_train, X_test, y_train, y_test, X_full_train, X_full_test = train_test_split(
    X_model,
    y_full,
    X_full,
    test_size=0.30,
    random_state=42,
    stratify=y_full
)

# save model-ready splits
X_train.to_csv("X_5_30_train.csv", index=False)
X_test.to_csv("X_5_30_test.csv", index=False)
y_train.to_csv("y_5_30_train.csv", index=False)
y_test.to_csv("y_5_30_test.csv", index=False)

# save full test data with context (traffic types, source_file, etc.)
X_full_test.to_csv("test_5_30_full_context.csv", index=False)

print("Saved train/test splits and full context test set")

# define models
models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(random_state=42))
    ]),

    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

results = []
best_model = None
best_f1 = -1
best_model_name = ""

# train and evaluate models
for name, pipeline in models.items():
    print(f"\n{name}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    # track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

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

# compare models
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv("model_comparison_congestion_5_30.csv", index=False)

print("\nModel comparison saved")

# save best model
joblib.dump(best_model, "best_congestion_model_5_30.pkl")

print(f"\nBest model: {best_model_name}")
print(f"Best F1 score: {best_f1}")
print("Saved best model to best_congestion_model.pkl")

# window size 1

# Features used for training:
# ['total_packets', 'total_bytes', 'mean_packet_size', 'max_packet_size', 'std_packet_size', 'active_flows', 'retransmissions', 'duplicate_acks', 'lost_segments', 'mean_rtt', 'max_rtt', 'std_rtt', 'mean_bytes_in_flight', 'max_bytes_in_flight', 'mean_window_size', 'tcp_packet_count', 'udp_packet_count', 'packets_per_sec', 'bytes_per_sec', 'tcp_ratio', 'udp_ratio', 'avg_bytes_per_flow', 'avg_packets_per_flow']
# Saved train/test splits and full context test set

# Logistic Regression
# Accuracy: 0.7393364928909952
# Precision: 0.7247706422018348
# Recall: 0.7596153846153846
# F1: 0.7417840375586855

# Decision Tree
# Accuracy: 0.8199052132701422
# Precision: 0.8113207547169812
# Recall: 0.8269230769230769
# F1: 0.819047619047619

# Extra Trees
# Accuracy: 0.8436018957345972
# Precision: 0.865979381443299
# Recall: 0.8076923076923077
# F1: 0.835820895522388

# KNN
# Accuracy: 0.7725118483412322
# Precision: 0.7857142857142857
# Recall: 0.7403846153846154
# F1: 0.7623762376237624

# SVM
# Accuracy: 0.7677725118483413
# Precision: 0.8160919540229885
# Recall: 0.6826923076923077
# F1: 0.743455497382199

# Gradient Boosting
# Accuracy: 0.8672985781990521
# Precision: 0.8454545454545455
# Recall: 0.8942307692307693
# F1: 0.8691588785046729

# XGBoost
# Accuracy: 0.8957345971563981
# Precision: 0.8727272727272727
# Recall: 0.9230769230769231
# F1: 0.897196261682243



# Best model: Random Forest
# Best F1 score: 0.9004739336492891

# Random Forest
# Accuracy: 0.9004739336492891
# Precision: 0.8878504672897196
# Recall: 0.9134615384615384
# F1: 0.9004739336492891

# Classification Report:
#                precision    recall  f1-score   support

# No Congestion       0.91      0.89      0.90       107
#    Congestion       0.89      0.91      0.90       104

#      accuracy                           0.90       211
#     macro avg       0.90      0.90      0.90       211
#  weighted avg       0.90      0.90      0.90       211

# Confusion Matrix:
# [[95 12]
#  [ 9 95]]


# window size 5

# Best model: Extra Trees
# Best F1 score: 0.9767441860465116

