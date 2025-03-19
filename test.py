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
from utilitiesENAS import *
from PlotterENAS import *
from ReportENAS import *
from globalsENAS import *
from configENAS import *

tecNAS = TECNAS()
arch = tecNAS.random_individual()
integer_vector = []

#ARCH TO INT FUNCTION. ALSO NEED TO CREATE INT TO ARCH FUNCTION
for dictio in arch.genotype.gen_list:
    for layer_type in dictio.keys():
        print(layer_type)
        if layer_type == 'INP':
            continue
        elif layer_type == 'CONV':
            nf = get_key_from_value(NUM_FILTERS, dictio[layer_type][0])
            integer_vector.append(nf)
            ks = get_key_from_value(CONV_KERNELS, dictio[layer_type][1])
            integer_vector.append(ks)
        elif layer_type == 'POOLMAX':
            ks = get_key_from_value(POOL_KERNELS, dictio[layer_type])
            integer_vector.append(ks)
        elif layer_type == 'FLATTEN':
            continue
        elif layer_type == 'DENSE':
            nn = get_key_from_value(DENSE_NEURONS, dictio[layer_type][0])
            integer_vector.append(nn)
            act = get_key_from_value(ACTIVATION_FUNCTIONS, dictio[layer_type][1])
            integer_vector.append(act)

print(integer_vector)
print(arch.genotype.gen_list)
        
#from dictio in arch.genotype:
#    get_key_from_value(dictio, val)

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
