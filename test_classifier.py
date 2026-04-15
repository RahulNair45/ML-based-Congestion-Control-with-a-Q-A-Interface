import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# 1. LOAD DATA
df = pd.read_csv("TimeBasedFeatures-Dataset-15s-NO-VPN.csv")

label_col = "class1"   # target col
X = df.drop(columns=[label_col]) # remove target so model dosent have acess to sol
y = df[label_col] # just target

# encode string labels
label_encoder = LabelEncoder() # converts text lables into num
y = label_encoder.fit_transform(y) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 2. DEFINE MODELS
# -------------------------
models = {
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fills missing numeric values with the column median
        ("model", DecisionTreeClassifier(
            max_depth=10,          # limits tree depth; lower = less overfitting, higher = more complex model
            min_samples_split=5,   # minimum samples needed to split a node; higher = smoother tree
            min_samples_leaf=2,    # minimum samples allowed in a leaf; helps reduce overfitting/noisy leaves
            criterion="gini",      # split quality measure; try "gini" or "entropy"
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fills missing values
        ("scaler", StandardScaler()),                   # scales features so distance-based KNN works properly
        ("model", KNeighborsClassifier(
            n_neighbors=5,      # number of neighbors used for voting; small = sensitive, large = smoother
            weights="distance", # closer neighbors count more than far ones; often improves KNN
            metric="minkowski", # distance metric family
            p=2                 # p=2 means Euclidean distance; try p=1 for Manhattan distance
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fills missing values
        ("scaler", StandardScaler()),                   # SVM is very sensitive to feature scale
        ("model", SVC(
            kernel="rbf",        # decision boundary type; "rbf" is strong default, try "linear" too
            C=1.0,               # regularization strength; larger = fits training data more closely
            gamma="scale",       # controls how local the boundary is; can strongly affect performance
            probability=True,    # enables probability/confidence outputs
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fills missing values
        ("model", XGBClassifier(
            n_estimators=550,        # number of boosting trees; more trees can improve accuracy up to a point
            max_depth=8,             # tree depth; lower = less overfitting, higher = more complex patterns
            learning_rate=0.19,       # step size for each tree; lower often generalizes better
            subsample=0.85,           # fraction of rows used per tree; helps reduce overfitting
            colsample_bytree=0.92,    # fraction of features used per tree; also helps reduce overfitting
            min_child_weight=1,      # minimum weight needed in a child node; larger = more conservative splits
            reg_lambda=1.0,          # L2 regularization; helps control overfitting
            eval_metric="mlogloss",  # loss used during training for multiclass problems
            random_state=42
        ))
    ])
}




# -------------------------
# 3. TRAIN + EVALUATE
# -------------------------
results = []

for name, pipeline in models.items():
    print(f"\n--- {name} ---")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc})

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# -------------------------
# 4. COMPARE MODELS
# -------------------------
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\nModel Comparison:")
print(results_df)


# n_estimators=200,        # number of boosting trees; more trees can improve accuracy up to a point
# max_depth=6,             # tree depth; lower = less overfitting, higher = more complex patterns
# learning_rate=0.1,       # step size for each tree; lower often generalizes better
# subsample=0.8,           # fraction of rows used per tree; helps reduce overfitting
# colsample_bytree=0.8,    # fraction of features used per tree; also helps reduce overfitting
# min_child_weight=1,      # minimum weight needed in a child node; larger = more conservative splits
# reg_lambda=1.0,          # L2 regularization; helps control overfitting
# eval_metric="mlogloss",  # loss used during training for multiclass problems
# random_state=42


# --- XGBoost ---
# Accuracy: 0.9520356943669828
#               precision    recall  f1-score   support

#     BROWSING       0.93      0.99      0.96       500
#         CHAT       0.94      0.88      0.91       178
#           FT       0.89      0.91      0.90       204
#         MAIL       0.88      0.84      0.86        50
#          P2P       0.98      0.97      0.98       200
#    STREAMING       0.93      0.79      0.85        96
#         VOIP       1.00      0.99      0.99       565

#     accuracy                           0.95      1793
#    macro avg       0.94      0.91      0.92      1793
# weighted avg       0.95      0.95      0.95      1793

# Model Comparison:
#            Model  Accuracy
# 3        XGBoost  0.952036
# 0  Decision Tree  0.904629
# 1            KNN  0.863358
# 2            SVM  0.716118

#_______________________________________________________________

# Find best paramaters

#XGBoost

# # XGBoost pipeline
# xgb_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("model", XGBClassifier(
#         eval_metric="mlogloss",
#         random_state=42
#     ))
# ])

# # parameter grid
# param_grid_xgb = {
#     "model__n_estimators": [100, 200, 300],
#     "model__max_depth": [3, 4, 6, 8],
#     "model__learning_rate": [0.01, 0.05, 0.1],
#     "model__subsample": [0.8, 1.0],
#     "model__colsample_bytree": [0.8, 1.0]
# }

# Best params: {'model__colsample_bytree': 1.0, 'model__learning_rate': 0.1, 'model__max_depth': 8, 'model__n_estimators': 300, 'model__subsample': 0.8}
# Best CV score: 0.9505523644357023
# Test Accuracy: 0.9503625209146681

# Best params: {'model__colsample_bytree': 0.9, 'model__learning_rate': 0.2, 'model__max_depth': 8, 'model__n_estimators': 500, 'model__subsample': 0.9}
# Best CV score: 0.9526110288443028
# Test Accuracy: 0.950920245398773

#Best params: {'model__colsample_bytree': 0.95, 'model__learning_rate': 0.2, 'model__max_depth': 8, 'model__n_estimators': 500, 'model__subsample': 0.85}
# Best CV score: 0.9526729633226131
# Test Accuracy: 0.9514779698828778

# Best params: {'model__colsample_bytree': 0.95, 'model__learning_rate': 0.19, 'model__max_depth': 8, 'model__n_estimators': 550, 'model__subsample': 0.85}
# Best CV score: 0.9528024928668503
# Test Accuracy: 0.9525934188510876

# Best params: {'model__colsample_bytree': 0.93, 'model__learning_rate': 0.19, 'model__max_depth': 8, 'model__n_estimators': 550, 'model__subsample': 0.85}
# Best CV score: 0.9528024928668503
# Test Accuracy: 0.9525934188510876

# Best params: {'model__colsample_bytree': 0.92, 'model__learning_rate': 0.19, 'model__max_depth': 8, 'model__n_estimators': 550, 'model__subsample': 0.85}
# Best CV score: 0.9528024928668503
# Test Accuracy: 0.9525934188510876

# param_grid_xgb = {
#     "model__n_estimators": [540, 550, 560],
#     "model__max_depth": [8],
#     "model__learning_rate": [0.19],
#     "model__subsample": [0.85],
#     "model__colsample_bytree": [0.91, 0.92]
# }

# # grid search
# grid_xgb = GridSearchCV(
#     estimator=xgb_pipeline,
#     param_grid=param_grid_xgb,
#     cv=5,
#     scoring="f1_weighted",
#     n_jobs=-1,
#     verbose=1
# )

# grid_xgb.fit(X_train, y_train)

# print("Best params:", grid_xgb.best_params_)
# print("Best CV score:", grid_xgb.best_score_)

# # best model
# best_xgb = grid_xgb.best_estimator_

# # test set evaluation
# y_pred = best_xgb.predict(X_test)

# print("Test Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# print(confusion_matrix(y_test, y_pred))

