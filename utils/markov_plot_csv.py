import os
import pandas as pd
os.add_dll_directory(r"C:\Program Files\Graphviz\bin")
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

import pygraphviz as pgv
from IPython.display import Image
import random
import numpy as np

df = pd.read_csv("matrix_CROSS.csv")

states = df.columns[1:]   # columnas destino
origins = df["CROSS"]     # estados origen

g = pgv.AGraph(directed=True)

# agregar nodos
for s in states:
    g.add_node(s)

# agregar transiciones
for i, row in df.iterrows():
    origin = row["CROSS"]
    for dest in states:
        prob = row[dest]
        if prob > 0.10:   # solo dibuja si hay transición
            g.add_edge(origin, dest, label=round(prob,3))

# layout más claro
g.node_attr.update(shape="circle", style="filled", fillcolor="lightblue")
g.edge_attr.update(fontsize="30")
g.graph_attr.update(rankdir="LR", overlap="false")
g.draw("markov_from_csv.png", prog="dot")

os.startfile("markov_from_csv.png")