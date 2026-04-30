import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# load aggregated window-level dataset
df = pd.read_csv("aggregated_dataset_1.csv")

# columns to drop before training because they are labels, metadata, or possible leakage
drop_cols = [
    "source_file",                 # tells model which capture the row came from, so it could memorize captures
    "time_window",                 # time/order identifier, not a real congestion feature
    "window_index",                # another time/order identifier that could leak capture position
    "traffic_browsing",            # traffic label is saved for analysis, not used to predict congestion
    "traffic_email_chat",          # traffic label is saved for analysis, not used to predict congestion
    "traffic_streaming",           # traffic label is saved for analysis, not used to predict congestion
    "traffic_file_transfer",       # traffic label is saved for analysis, not used to predict congestion
    "traffic_voice_video_call"     # traffic label is saved for analysis, not used to predict congestion
]

# target column that the model is trying to predict
label_col = "congestion_label"

# keep all non-target columns so we can still use traffic/source info later for analysis
X_full = df.drop(columns=[label_col])

# y contains the true congestion labels
y = df[label_col]

# model input only uses congestion-related numerical features
X_model = X_full.drop(columns=drop_cols, errors="ignore")

# load saved best model as the model configuration
model = joblib.load("best_congestion_model_1.pkl")

# show which features are actually being fed into the model
print("Features used for evaluation:")
print(X_model.columns.tolist())

# run 5-fold cross-validation to estimate performance across multiple splits
cv_results = cross_validate(
    model,
    X_model,
    y,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=False
)

# print average CV performance across all folds
print("\nCross-validation performance:")
print("Accuracy mean:", np.mean(cv_results["test_accuracy"]))
print("Precision mean:", np.mean(cv_results["test_precision"]))
print("Recall mean:", np.mean(cv_results["test_recall"]))
print("F1 mean:", np.mean(cv_results["test_f1"]))

# print standard deviation to show how stable the model is across folds
print("\nCross-validation stability:")
print("Accuracy std:", np.std(cv_results["test_accuracy"]))
print("Precision std:", np.std(cv_results["test_precision"]))
print("Recall std:", np.std(cv_results["test_recall"]))
print("F1 std:", np.std(cv_results["test_f1"]))

# save fold-by-fold CV results so they can be used in the report
pd.DataFrame(cv_results).to_csv("cv_results_best_model_1.csv", index=False)

# get out-of-fold predictions for every row
# each row is predicted by a model that did not train on that row
y_pred_cv = cross_val_predict(
    model,
    X_model,
    y,
    cv=5
)

# attach true labels and CV predictions to the full dataset
df["y_true"] = y
df["y_pred_cv"] = y_pred_cv

# evaluate overall performance using out-of-fold predictions
print("\nOverall performance from cross-validated predictions:")
print("Accuracy:", accuracy_score(df["y_true"], df["y_pred_cv"]))
print("Precision:", precision_score(df["y_true"], df["y_pred_cv"], zero_division=0))
print("Recall:", recall_score(df["y_true"], df["y_pred_cv"], zero_division=0))
print("F1:", f1_score(df["y_true"], df["y_pred_cv"], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(df["y_true"], df["y_pred_cv"]))

# traffic columns used to break down performance by traffic type
traffic_cols = [
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

# store traffic-type performance rows
traffic_results = []

print("\nPerformance by traffic type:")

# loop through each traffic type and evaluate only windows where that traffic appears
for col in traffic_cols:
    if col not in df.columns:
        continue

    # only evaluate rows where this traffic type is present
    subset = df[df[col] == 1]

    # skip traffic types that do not appear in this dataset
    if len(subset) == 0:
        continue

    # true and predicted labels for this traffic type
    y_true_sub = subset["y_true"]
    y_pred_sub = subset["y_pred_cv"]

    # calculate metrics for this traffic type
    acc = accuracy_score(y_true_sub, y_pred_sub)
    precision = precision_score(y_true_sub, y_pred_sub, zero_division=0)
    recall = recall_score(y_true_sub, y_pred_sub, zero_division=0)
    f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)

    # save metrics for this traffic type
    traffic_results.append({
        "Traffic Type": col,
        "Samples": len(subset),
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    # print metrics for this traffic type
    print(f"\n{col}")
    print("Samples:", len(subset))
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

# save traffic type results sorted by F1
traffic_df = pd.DataFrame(traffic_results).sort_values("F1", ascending=False)
traffic_df.to_csv("traffic_type_cv_1.csv", index=False)

print("\nSaved traffic-type CV performance to traffic_type_cv_1.csv")

# count how many traffic types are active in each window
df["num_traffic_types"] = df[traffic_cols].sum(axis=1)

# windows with exactly one traffic type active
single_traffic = df[df["num_traffic_types"] == 1]

# windows with more than one traffic type active
mixed_traffic = df[df["num_traffic_types"] > 1]

# store single vs mixed traffic results
mixed_results = []

# helper function to evaluate any subset of rows
def evaluate_subset(name, subset):
    # skip empty groups
    if len(subset) == 0:
        return

    # calculate metrics for this subset
    acc = accuracy_score(subset["y_true"], subset["y_pred_cv"])
    precision = precision_score(subset["y_true"], subset["y_pred_cv"], zero_division=0)
    recall = recall_score(subset["y_true"], subset["y_pred_cv"], zero_division=0)
    f1 = f1_score(subset["y_true"], subset["y_pred_cv"], zero_division=0)

    # save metrics for this subset
    mixed_results.append({
        "Group": name,
        "Samples": len(subset),
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    # print metrics for this subset
    print(f"\n{name}")
    print("Samples:", len(subset))
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

# compare whether single-traffic windows are easier than mixed-traffic windows
print("\nSingle vs mixed traffic performance:")
evaluate_subset("Single Traffic Type", single_traffic)
evaluate_subset("Mixed Traffic Types", mixed_traffic)

# save mixed vs single traffic results
pd.DataFrame(mixed_results).to_csv("mixed_vs_single_cv_1.csv", index=False)

# save full dataset with CV predictions attached for later error analysis
df.to_csv("dataset_with_cv_1.csv", index=False)

print("\nSaved mixed/single traffic results to mixed_vs_single_cv_1.csv")
print("Saved dataset with CV predictions to dataset_with_cv_1.csv")

# train final model on all data for future use
# this is for deployment/reuse, not for evaluation
# final_model = model.fit(X_model, y)
# joblib.dump(final_model, "final_congestion_model.pkl")

# print("\nSaved final model trained on all data to final_congestion_model.pkl")