gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

#TRY CROSSOVER-NONE and MUTATION-NONE
#CHECK REPORTS AND PLOTS SINCE I ADDED BESTGEN IN THE REPORT
#FOR PLOTS, CHANGE NAME OF CROSSOVER AND MUTATION OPS FOR LABELS
#REPORT FOR GAs PERFORMANCE: Right now it takes the last of each generation and sum them. Next time, its only the last number per EXECUTION.
#ALSO, CHECK REPORTS FOR RANDOM
#SUCCES MUT AND CROSS ITS A RATIO. I NEED THE TOTAL GENETIC OPERATIONS, CHECK IF THEY ARE BEING SAVED IN THE REPORT (UC GOT MORE THAN 1, WTF)
#VALIDATE_ARCHITECTURE HAS TO BE PERFECT
#LIMIT NUMBER OF POOLING LAYERS

import os
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import subprocess
import sys

def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_running_in_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    get_ipython().system(f"pip install colorama")
    os.chdir('/content/drive/MyDrive/DCC/Research/Layer_article')


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
print(f'{tf.__version__=}')
tf.get_logger().setLevel('ERROR')

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))
tecNAS = TECNAS(regressor_type = 1)
tecNAS.ENAS()