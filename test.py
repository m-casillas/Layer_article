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
from utilitiesENAS import *
from PlotterENAS import *
from ReportENAS import *
from globalsENAS import *
from configENAS import *

tecNAS = TECNAS()
reporter = ReportENAS()
arch = tecNAS.random_individual()
archP1 = tecNAS.random_individual()
archP2 = tecNAS.random_individual()
archM = tecNAS.random_individual()

print(arch)
print(arch.integer_encoding)

arch.isChild = True
archM.isChild = True
archM.parent1 = archP1
archM.parent2 = archP2
archM.before_mutation = copy.deepcopy(arch)
archM.dP1 = hamming_distance(arch.integer_encoding, archP1.integer_encoding)
archM.dP2 = hamming_distance(arch.integer_encoding, archP2.integer_encoding)
archM.dBM = hamming_distance(archM.integer_encoding, arch.integer_encoding)
reporter.save_arch_info(archM)
reporter.save_arch_info(archP1)
reporter.save_arch_info(archP2)


