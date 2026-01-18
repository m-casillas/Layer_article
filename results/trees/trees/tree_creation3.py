import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# 1. Cargar y limpiar datos
# =========================
df = pd.read_csv("merged.csv")
df = df.dropna(subset=["Improved"])

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
# 4. Espacio evolutivo
# (hiperparámetros RF)
# =========================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}

base_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# =========================
# 5. Búsqueda tipo NAS
# =========================
search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_rf = search.best_estimator_

# =========================
# 6. Evaluación real
# =========================
y_pred = best_rf.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)

print("Mejores hiperparámetros:", search.best_params_)
print(f"Validation Accuracy: {val_acc:.4f}")

# =========================
# 7. Guardar mejor modelo
# =========================
joblib.dump(best_rf, "operator_forest.pkl")
joblib.dump(le, "operator_encoder.pkl")
