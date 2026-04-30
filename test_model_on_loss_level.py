import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# aggregated packet-loss test dataset
test_file = "pLoss_aggregated_3.csv"

# saved congestion models to test
model_files = [
    "best_congestion_model_3_20.pkl",
    "best_congestion_model_3_30.pkl",
]

# output files
overall_output_file = "pLoss_model_overall_performance_3.csv"
loss_level_output_file = "pLoss_model_performance_by_loss_level_3.csv"

# load full pLoss test data
df_full = pd.read_csv(test_file)

# true binary congestion label
label_col = "congestion_label"
y_true = df_full[label_col]

# columns to remove before feeding data into the model
# packet_loss_level is removed because old saved models were not trained with it
drop_cols = [
    "source_file",
    "time_window",
    "window_index",
    "congestion_label",
    "packet_loss_level",
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

# model input only contains the same congestion-related features used during training
X = df_full.drop(columns=drop_cols, errors="ignore")

overall_results = []
loss_level_results = []

for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"\nSkipping missing model: {model_file}")
        continue

    print(f"\nTesting model: {model_file}")

    # load saved model
    model = joblib.load(model_file)

    # match test features to the exact features the model was trained on
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        X_model = X.reindex(columns=expected_cols, fill_value=0)
    else:
        X_model = X.copy()

    # predict congestion using only model features
    y_pred = model.predict(X_model)

    # overall binary congestion performance
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    overall_results.append({
        "Model": model_file,
        "Samples": len(df_full),
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    print("\nOverall Performance:")
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["No Congestion", "Congestion"],
        zero_division=0
    ))

    # copy full data so packet_loss_level is available only for analysis
    results_df = df_full.copy()
    results_df["y_true"] = y_true
    results_df["y_pred"] = y_pred

    print("\nPerformance by packet loss level:")

    for loss_level in sorted(results_df["packet_loss_level"].dropna().unique()):
        subset = results_df[results_df["packet_loss_level"] == loss_level]

        if len(subset) == 0:
            continue

        y_true_sub = subset["y_true"]
        y_pred_sub = subset["y_pred"]

        sub_acc = accuracy_score(y_true_sub, y_pred_sub)
        sub_precision = precision_score(y_true_sub, y_pred_sub, zero_division=0)
        sub_recall = recall_score(y_true_sub, y_pred_sub, zero_division=0)
        sub_f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)

        loss_level_results.append({
            "Model": model_file,
            "Packet Loss Level": loss_level,
            "Samples": len(subset),
            "Accuracy": sub_acc,
            "Precision": sub_precision,
            "Recall": sub_recall,
            "F1": sub_f1
        })

        print(f"\nPacket loss level: {loss_level}")
        print("Samples:", len(subset))
        print("Accuracy:", sub_acc)
        print("Precision:", sub_precision)
        print("Recall:", sub_recall)
        print("F1:", sub_f1)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true_sub, y_pred_sub))

# save result tables
overall_df = pd.DataFrame(overall_results)
loss_level_df = pd.DataFrame(loss_level_results)

overall_df.to_csv(overall_output_file, index=False)
loss_level_df.to_csv(loss_level_output_file, index=False)

print(f"\nSaved overall model results to {overall_output_file}")
print(f"Saved loss-level model results to {loss_level_output_file}")