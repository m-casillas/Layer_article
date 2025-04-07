import copy
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time


from Genotype import *
from Architecture import *
from LayerRepresentation import *
from Mutator import *
from Crossover import *
from TECNAS import *
from PlotterENAS import *
from ReportENAS import *
from globalsENAS import *
from configENAS import *

os.system("cls")

tecnas = TECNAS()
arch = tecnas.random_individual()
print(arch.genotype.gen_list)
print()
print(f'{MUTABLE_LCHANGETYPE_INDEXES = }')
print(f'{MUTABLE_LCHANGEPARAM_INDEXES = }')
print(f'{SPC_INDEXES = }')

#     0               1                      2                   3                      4                  5                6                           7
#[{'INP': 32}, {'CONV': [256, 5]}, {'POOLMAX': [-1, 3]}, {'CONV': [128, 5]}, {'POOLMAX': [-1, 3]}, {'FLATTEN': None}, {'DENSE': [128, 'relu']}, {'DENSE': [10, 'softmax']}]