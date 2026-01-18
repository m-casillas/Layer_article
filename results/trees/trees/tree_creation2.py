import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# 1. Cargar y limpiar datos
# =========================
df = pd.read_csv("merged.csv")
df = df.dropna(subset=["Improved"])

state_features = [
    "Progress",
    "HD_PB",
    "HD_P1",
    "HD_P2",
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
# 4. Definir espacio evolutivo
# (hiperparámetros)
# =========================
param_grid = {
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

base_tree = DecisionTreeClassifier(random_state=42)

# =========================
# 5. Búsqueda tipo NAS
# =========================
search = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,                # fitness robusto
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_tree = search.best_estimator_

# =========================
# 6. Evaluación real
# =========================
y_pred = best_tree.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)

print("Mejores hiperparámetros:", search.best_params_)
print(f"Validation Accuracy: {val_acc:.4f}")

# =========================
# 7. Guardar mejor modelo
# =========================
joblib.dump(best_tree, "operator_tree.pkl")
joblib.dump(le, "operator_encoder.pkl")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(
    best_tree,
    feature_names=state_features,
    class_names=le.classes_,  # <-- 'A', 'B'
    filled=True,
    rounded=True
)
plt.show()
