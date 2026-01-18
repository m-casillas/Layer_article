import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("merged.csv")
df = df.dropna(subset=['Improved'])
state_features = [
    "Progress",
    "HD_PB",
    "HD_P1",
    "HD_P2",
    "dHV"
]

df = df.drop_duplicates(subset=state_features)

X = df[state_features]
y = df["Improved"]   # SPC_MPAR, UC_MSWAP, etc.

le = LabelEncoder()
y_encoded = le.fit_transform(y)
tree = DecisionTreeClassifier(
    max_depth=4,        # interpretabilidad
    min_samples_leaf=10,
    random_state=42
)

tree.fit(X, y_encoded)

joblib.dump(tree, "operator_tree.pkl")
joblib.dump(le, "operator_encoder.pkl")

'''
from sklearn.tree import export_text
tree_rules = export_text(tree, feature_names=state_features)
print(tree_rules)
'''

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=state_features,
    class_names=le.classes_,  # <-- 'A', 'B'
    filled=True,
    rounded=True
)
plt.show()

'''
#Prediccion
tree = joblib.load("operator_tree.pkl")
le = joblib.load("operator_encoder.pkl")


current_state = {
    "Accuracy":0.7548,
    "HD_PB":1,
    "HD_P1":0.1,
    "HD_P2":0.2
}

X_state = pd.DataFrame([current_state])
predicted_class = tree.predict(X_state)[0]
operator = le.inverse_transform([predicted_class])[0]
print(operator)

'''