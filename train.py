import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
os.makedirs("models", exist_ok=True)
# 1. Load Dataset
print("Loading dataset.csv...")
df = pd.read_csv("dataset.csv")
# 2. Data Preprocessing: Duplicates & Missing Values
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows.")
base_features = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
target_col = 'Potability'
# Fill missing values with median if any exist
for col in base_features:
    df[col] = df[col].fillna(df[col].median())
# Outlier Detection & Handling (IQR Winsorization / Capping)
for col in base_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)
# 3. Feature Engineering
def engineer_features(data_df):
    X_eng = data_df[base_features].copy()
    # Deviation from neutral pH (7.0)
    X_eng['ph_neutral_diff'] = np.abs(X_eng['ph'] - 7.0)
    # Ratio of dissolved solids to conductivity
    X_eng['Solids_Conductivity_ratio'] = X_eng['Solids'] / (X_eng['Conductivity'] + 1e-5)
    # Disinfectant to mineral ratio
    X_eng['Chloramines_Sulfate_ratio'] = X_eng['Chloramines'] / (X_eng['Sulfate'] + 1e-5)
    # Combined chemical burden index
    X_eng['Total_Chemical_Load'] = (
        X_eng['Chloramines'] + X_eng['Organic_carbon'] + (X_eng['Trihalomethanes'] / 10.0)
    )
    return X_eng

X = engineer_features(df)
y = df[target_col]

# 4. Feature Selection (Mutual Information)
selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X, y)
feature_scores = dict(zip(X.columns, np.round(selector.scores_, 4)))
print("Feature Importance (Mutual Info):", feature_scores)

# 5. Train-Test Split (Stratified) & Feature Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Advanced ML Model Training
models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
}

evaluation_results = {}
best_model_name = None
best_f1 = -1.0
best_model_obj = None

print("\n--- Model Evaluation Results ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred) * 100, 2)
    rec = round(recall_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)
    auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
    cm = confusion_matrix(y_test, y_pred).tolist()

    evaluation_results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Confusion_Matrix": cm
    }
    print(f"{name:20s} | Acc: {acc}% | Prec: {prec}% | Rec: {rec}% | F1: {f1}% | ROC-AUC: {auc}%")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model
# 7. Save Best Model, Scaler & Metrics
joblib.dump(best_model_obj, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
metadata = {
    "best_model": best_model_name,
    "total_records": len(df),
    "potable_count": int((y == 1).sum()),
    "not_potable_count": int((y == 0).sum()),
    "feature_scores": feature_scores,
    "metrics": evaluation_results
}

with open("models/metrics.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nBest Model '{best_model_name}' (F1: {best_f1}%) saved successfully in models/!")