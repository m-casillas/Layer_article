import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeRegressor, plot_tree

# -----------------------------
# Cargar datos
# -----------------------------
df = pd.read_csv("merged.csv")

# Convertir Integer_encoding de string a lista
df["Integer_encoding"] = df["Integer_encoding"].apply(ast.literal_eval)

X = np.array(df["Integer_encoding"].tolist())
y = df["Accuracy"].values

# Normalizar Accuracy solo para color
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

# -----------------------------
# 1. Reducción de dimensionalidad (PCA + t-SNE)
# -----------------------------

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=20)
plt.colorbar(label="Accuracy")
plt.title("PCA 2D coloreado por Accuracy")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# t-SNE 2D (más lento pero más expresivo)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", s=20)
plt.colorbar(label="Accuracy")
plt.title("t-SNE 2D coloreado por Accuracy")
plt.show()

# -----------------------------
# 2. Heatmap de vectores + Accuracy
# -----------------------------

# Crear DataFrame expandido
vec_len = X.shape[1]
vec_cols = [f"pos_{i}" for i in range(vec_len)]
df_vec = pd.DataFrame(X, columns=vec_cols)
df_vec["Accuracy"] = y

# Ordenar por Accuracy descendente
df_vec_sorted = df_vec.sort_values("Accuracy", ascending=False)

plt.figure(figsize=(14, 8))
sns.heatmap(
    df_vec_sorted[vec_cols],
    cmap="tab20",
    cbar=True
)
plt.title("Heatmap de Integer_encoding (ordenado por Accuracy)")
plt.xlabel("Posición en el vector")
plt.ylabel("Vectores (ordenados)")
plt.show()

# Accuracy como columna aparte
plt.figure(figsize=(2, 8))
sns.heatmap(
    df_vec_sorted[["Accuracy"]],
    cmap="viridis",
    cbar=True
)
plt.title("Accuracy")
plt.show()

# -----------------------------
# 3. Coordenadas paralelas
# -----------------------------

# Discretizar Accuracy en rangos
df_pc = df_vec.copy()
df_pc["acc_bin"] = pd.qcut(df_pc["Accuracy"], q=3, labels=["baja", "media", "alta"])

plt.figure(figsize=(14, 6))
parallel_coordinates(
    df_pc[vec_cols + ["acc_bin"]],
    class_column="acc_bin",
    colormap=plt.cm.coolwarm,
    alpha=0.3
)
plt.title("Coordenadas paralelas agrupadas por Accuracy")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4. Boxplots por categoría (por posición)
# -----------------------------

pos = 0  # cambia la posición que quieras analizar
df_box = pd.DataFrame({
    "categoria": X[:, pos],
    "Accuracy": y
})

plt.figure(figsize=(10, 5))
sns.boxplot(x="categoria", y="Accuracy", data=df_box)
plt.title(f"Accuracy por categoría en posición {pos}")
plt.xticks(rotation=90)
plt.show()

# -----------------------------
# 5. Árbol de decisión (interpretabilidad)
# -----------------------------

tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=30,
    random_state=42
)
tree.fit(X, y)

plt.figure(figsize=(20, 8))
plot_tree(
    tree,
    feature_names=vec_cols,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Árbol de decisión: patrones del vector → Accuracy")
plt.show()
