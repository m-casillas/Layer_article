import os
os.add_dll_directory(r"C:\Program Files\Graphviz\bin")
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

import pygraphviz as pgv
from IPython.display import Image
import random
import numpy as np

states = ["A", "B", "C"]

g = pgv.AGraph(directed=True)

# agregar nodos
for s in states:
    g.add_node(s)

# agregar transiciones (incluyendo hacia sí mismo)
for s1 in states:
    
    probs = np.random.dirichlet(np.ones(len(states)))
    
    for s2, p in zip(states, probs):
        g.add_edge(s1, s2, label=round(p, 2))

g.draw("markov.png", prog="dot")
os.startfile("markov.png")