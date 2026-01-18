import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# 1. Cargar y limpiar datos
# =========================
df = pd.read_csv("merged.csv")
df = df.dropna(subset=["Improved"])
df = df[df["Improved"] != "NONE"].copy()


state_features = [
    "HD_PB",
    "dHV"
]

df = df.drop_duplicates(subset=state_features)

X = df[state_features]
y = df["Improved"]

# =========================
# 2. Codificar etiquetas
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# =========================
# 3. Train / Validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# =========================
# 4. Modelo base XGBoost
# =========================
base_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1          # 🔥 IMPORTANTE para evitar sobrecarga
)

# =========================
# 5. Espacio de búsqueda (Random)
# =========================
param_distributions = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "min_child_weight": [1, 5, 10],
    "gamma": [0, 0.5, 1.0]
}

# =========================
# 6. Randomized Search
# =========================
search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_distributions,
    n_iter=30,            # 🔥 puedes bajar a 20 si quieres aún más rápido
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_xgb = search.best_estimator_

# =========================
# 7. Evaluación
# =========================
y_pred = best_xgb.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)

print("Mejores hiperparámetros:", search.best_params_)
print(f"Validation Accuracy: {val_acc:.4f}")

# =========================
# 8. Guardar modelo y encoder
# =========================
joblib.dump(best_xgb, "operator_xgboost.pkl")
joblib.dump(le, "operator_encoder.pkl")

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    log_loss, roc_auc_score, cohen_kappa_score,
    confusion_matrix, classification_report
)

y_pred = best_xgb.predict(X_val)
y_proba = best_xgb.predict_proba(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 weighted:", f1_score(y_val, y_pred, average="weighted"))
print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))
print("Log Loss:", log_loss(y_val, y_proba))
print("ROC AUC:", roc_auc_score(
    y_val, y_proba, multi_class="ovr", average="weighted"
))
print("Cohen Kappa:", cohen_kappa_score(y_val, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_pred))
