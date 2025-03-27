#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]


#If mutation gives the same architecture, try another mutation
#Save epochs into architectures.csv
#Save the best architectures in other file. architectures.csv is for the surrogate model

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

"""#TESTING"""

'''path = '/content/drive/MyDrive/DCC/Materias/Semestre 02/Evolutive Computation/'
dir_results = 'final_project_results'
path_results = os.path.join(path, dir_results)
print(path_results)

MAIN_NPOP = 10
MUT_PROB = 1/5 #5 is the number of layers
GENERATIONS = 10
EXECUTIONS = 5 #1 execution is after #GENERATIONS
tecNAS = TECNAS()
tecNAS.ENAS()
'''

"""#MAIN"""

#random.seed(42)
tecNAS = TECNAS()
tecNAS.ENAS()

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
