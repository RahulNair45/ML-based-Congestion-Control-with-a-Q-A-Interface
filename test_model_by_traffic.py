import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load model and test data
model = joblib.load("best_congestion_model_5.pkl")

X_test = pd.read_csv("X_5_test.csv")
y_test = pd.read_csv("y_5_test.csv").values.ravel()
context = pd.read_csv("test_5_full_context.csv")

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
results_df.to_csv("traffic_type_5_performance.csv", index=False)

print("\nSaved traffic type performance to traffic_type_performance.csv")

# results for window size 1:

# Overall Performance:
# Accuracy: 0.9004739336492891
# Precision: 0.8878504672897196
# Recall: 0.9134615384615384
# F1: 0.9004739336492891

# ===== Performance by Traffic Type =====

# traffic_browsing
# Samples: 115
# Accuracy: 0.9217391304347826
# Precision: 0.890625
# Recall: 0.9661016949152542
# F1: 0.926829268292683

# traffic_email_chat
# Samples: 83
# Accuracy: 0.9518072289156626
# Precision: 0.9230769230769231
# Recall: 0.972972972972973
# F1: 0.9473684210526315

# traffic_streaming
# Samples: 71
# Accuracy: 0.9859154929577465
# Precision: 0.9736842105263158
# Recall: 1.0
# F1: 0.9866666666666667

# traffic_file_transfer
# Samples: 40
# Accuracy: 1.0
# Precision: 1.0
# Recall: 1.0
# F1: 1.0

# traffic_voice_video_call
# Samples: 65
# Accuracy: 0.8461538461538461
# Precision: 0.8823529411764706
# Recall: 0.8333333333333334
# F1: 0.8571428571428571