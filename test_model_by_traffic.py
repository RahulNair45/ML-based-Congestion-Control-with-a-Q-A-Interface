import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load model and test data
model = joblib.load("best_congestion_model_1.pkl")

X_test = pd.read_csv("X_1_test.csv")
y_test = pd.read_csv("y_1_test.csv").values.ravel()
context = pd.read_csv("test_1_full_context.csv")

# generate predictions
y_pred = model.predict(X_test)

# attach predictions to context
context["y_true"] = y_test
context["y_pred"] = y_pred

print("\nOverall Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))

# traffic columns
traffic_cols = [
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

results = []

print("\n===== Performance by Traffic Type =====")

for col in traffic_cols:
    if col not in context.columns:
        continue

    subset = context[context[col] == 1]

    if len(subset) == 0:
        continue

    y_true_sub = subset["y_true"]
    y_pred_sub = subset["y_pred"]

    acc = accuracy_score(y_true_sub, y_pred_sub)
    precision = precision_score(y_true_sub, y_pred_sub, zero_division=0)
    recall = recall_score(y_true_sub, y_pred_sub, zero_division=0)
    f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)

    results.append({
        "Traffic Type": col,
        "Samples": len(subset),
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    print(f"\n{col}")
    print("Samples:", len(subset))
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

# save results
results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv("traffic_type_performance.csv", index=False)

print("\nSaved traffic type performance to traffic_type_performance.csv")