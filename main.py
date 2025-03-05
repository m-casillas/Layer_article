'''
# TODO:

*   Implement binary encoding.
*   Save architectures information with corresponding PIs into files.
*   Implement Wilcoxon Test.

# CONSIDERATIONS:
* Binary and integer encodings will somehow be the same. How can I make the GOs different?
* What test should I use to find or not significant differences through different procedures? (Like ANOVA, but non-parametric)
'''

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

random.seed(42)
tecNAS = TECNAS()
if RUN_ENAS == True:
    tecNAS.ENAS()
else:
    plotter = PlotterENAS(tecNAS)
    bar_columns_all = ['Acc_mean', "Loss_mean", "CPU_hrs_mean", "Num_params_mean", "FLOPs_mean"]
    box_columns_all = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs"]

    bar_columns = ["FLOPs_mean"]
    box_columns = ["FLOPs"]
    plotter.create_grouped_plots(tecNAS.general_report_filenames, bar_columns, box_columns)
    #tecNAS.plot_accuracy_loss_fromFiles(tecNAS.medians_report_filenames, 'A')
    #tecNAS.plot_accuracy_loss_fromFiles(tecNAS.medians_report_filenames, 'L')

#TODO Change titles. Add Wilcoxon