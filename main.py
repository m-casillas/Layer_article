#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

#CHECK THE TOTAL ARCHS COUNT
#Add the succesfulCross and mut counters!
#LIMIT NUMBER OF POOLING LAYERS
#Save the best architectures in other file. architectures.csv is for the surrogate model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
os.system("cls")
print(f'{tf.__version__=}')
tf.get_logger().setLevel('ERROR')

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))
tecNAS = TECNAS()
tecNAS.ENAS()

'''
gen_list = [{'INP':28}, {'CONV':[256,7]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLAVG':[-1,3]}, {'FLATTEN':None}, {'DENSE':[256,'relu']}, {'DENSE':[10,'softmax']}]
gen_obj = Genotype('X','X',gen_list)
arch_obj = LayerRepresentation('S', 0, gen_obj)
idx = 6
print(arch_obj.genotype.gen_list)
print()
mut = Mutator()
print(arch_obj.genotype.gen_list[idx])
arch_obj.genotype.gen_list[idx], ltype = mut.mutate_layer_parameters(arch_obj.genotype, idx)
print(arch_obj.genotype.gen_list[idx])
print()
print(arch_obj.genotype.gen_list)
'''