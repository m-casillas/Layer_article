from colorama import Fore, Back, Style, init
init(autoreset=True)
print(Fore.RED + "Importing TensorFlow and other libraries..." + Style.RESET_ALL)
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import psutil
import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import regularizers
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Input
from keras.src import datasets, layers, models

#from keras.src import backend as K
#from tensorflow.keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split
tf.keras.mixed_precision.set_global_policy("mixed_float16")

