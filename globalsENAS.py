import os 
import sys, io
import itertools
import tensorflow as tf
import numpy as np
import random
import platform
import contextlib
import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.layers import Input
from datetime import datetime

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")


def list_to_dictionary(list1):
     return {i:value for i,value in enumerate(list1)}
     
def generate_letter_list():
    #Generate letters from A to ZZZZ, for the architecture idx
    letters = []
    for length in range(1, 6):  # Generate from length 1 to 4
        for combo in itertools.product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=length):
            letters.append("".join(combo))
    return letters

def find_median(pop_list):
    sorted_array = sorted(pop_list, key=lambda arch:arch.acc)
    n = len(sorted_array)
    median_arch = sorted_array[n // 2]  # Odd length, middle element
    idx = [i for i, arch in enumerate(sorted_array) if arch.acc == median_arch.acc]
    return median_arch, idx

def calculate_model_flops(model):
    if not model.built:
        model.build(input_shape=(None,) + model.input_shape[1:])
        
    tf.get_logger().setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:], dtype=tf.float32)]
    )
    concrete_func = forward_pass.get_concrete_function()
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    # Run profiler (output still printed by TF, can't be fully silenced)
    options = ProfileOptionBuilder.float_operation()
    prof = profile(frozen_func.graph, options=options)
    return prof.total_float_ops // 2  # Divide by 2 if TF counts multiply-adds as 2 FLOPs

def calculate_model_params(model):
    trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    return total_params

def create_pool_max_layer(ks = None):
        #A Pool layer has: kernel size (ks)... more to come
        #If ks is None, it is randomly created
        if ks == None:
            ks = random.choice(list(POOL_KERNELS.values()))
        return {'POOLMAX':[-1, ks]} #-1 is neccesary to keep pool layers the same size as conv layers. It will never mutate

def create_pool_avg_layer(ks = None):
        #A Pool layer has: kernel size (ks)... more to come
        #If ks is None, it is randomly created
        if ks == None:
            ks = random.choice(list(POOL_KERNELS.values()))
        return {'POOLAVG':[-1, ks]} #-1 is neccesary to keep pool layers the same size as conv layers. It will never mutate

def create_conv_layer(nf = None, ks = None):
        #A Conv layer has: Number of filters (nf) kernel size (ks)... more to come
        #If nf and ks are None, it is randomly created
        if nf == None and ks == None:
            #{'CONV':[NUM_FILTERS[np.random.randint(0, len(NUM_FILTERS))],CONV_KERNELS[np.random.randint(0, len(CONV_KERNELS))]]}, This is for MUTATION
            nf = random.choice(list(NUM_FILTERS.values())) 
            ks = random.choice(list(CONV_KERNELS.values()))
        return {'CONV':[nf, ks]}

def create_dense_layer(nn = None, act = None):
        #A Dense layer has: Number of neurons (nn) and activation function (act)... more to come
        #If nn and act are None, it is randomly created
        if nn == None and act == None:
            nn = random.choice(list(DENSE_NEURONS.values())[1:]) #Ignore the first number (10), that's for the last layer
            act = 'relu'
        return {'DENSE':[nn, act]}

def create_last_name(column_list):
    last_name = ''
    for c in column_list:
            last_name = last_name + c + '_'
    return last_name

def determine_label_filename(filename):
    if 'L_MODIFY_PARAMS' in filename:
        label = 'L_MODIFY_PARAMS'
    elif 'L_CHANGE_TYPE' in filename:
        label = 'L_CHANGE_TYPE'
    elif 'NONE' in filename:
        label = 'RANDOM'
    else:
        label = '?'
    return label

def get_key_from_value(dictio, val):
    #Get the key from a value in a dictionary
    key = next((k for k, v in dictio.items() if v == val), None)
    return key

def hamming_distance(str1, str2):
    #Calculate the Hamming distance between two strings. (integer, binary vectors, etc.)
    #It returns how many characters differ between two strings.
    #-1 means the architecture doesnot have a hamming distance (i.e. a children and its mutation, if it didn't mutate)
    #-2 means the vectors were not the same size.
    if len(str1) != len(str2):
        print('\nHamming distance error: Vectors must be of the same length')
        print(f'{str1 = }')
        print(f'{str2 = }')
        print('====================================\n')
        return -2
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def calculate_all_hamming_distances(arch_obj, mutated = False):
    #Mutated is used to check if the architecture was mutated or not. If it was, calculate the hamming distance between itself and the architecture before mutation.
    
    if mutated == True:
        arch_obj.dP1 = -1
        arch_obj.dP2 = -1
        arch_obj.dBM = hamming_distance(arch_obj.integer_encoding, arch_obj.before_mutation.integer_encoding)
    else:
        arch_obj.dP1 = hamming_distance(arch_obj.integer_encoding, arch_obj.parent1.integer_encoding)
        arch_obj.dP2 = hamming_distance(arch_obj.integer_encoding, arch_obj.parent2.integer_encoding)

         
def is_None_or_empty(object):
    #Check if an object is None or empty
    if object == None or object == []:
        return True
    return False

def check_within_bounds(number, LB, UB):
    #Used for keeping indexes that are going to change within the bounds
    if number > UB:
        number = LB
    return number

def select_type_filtering(lst, op_type):
    #Returns a random string from lst, excluding op_type.
    #Used for mutation by selecting an operation, excluding the current one.
    #For example, if op_type is CONV, it returns another layer type, like POOLMAX or POOLAVG
    filtered_list = [item for item in lst if item != op_type]
    return random.choice(filtered_list) if filtered_list else None

def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
#                0              1                2               3            4                  5                 6                       7
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#               -8              -7               -6              -5           -4                 -3                -2                      -1

if is_running_in_colab():
    print("Running in Google Colab.")
    path =  '/content/drive/MyDrive/DCC/Research/Layer_article'
elif platform.system() == 'Windows':
     print("Running in Windows.")
     path =  os.getcwd()
else:
     print("Running in Azure VM.")
     path = '/home/Super-IR/Thesis_code/Tesis_code/Layer_article'

#                        0      1         2        3            4        5
layer_types_list =    ['INP', 'CONV', 'POOLMAX', 'POOLAVG', 'FLATTEN', 'DENSE']
type_mutable_layers = ['CONV','POOLMAX']#,'POOLAVG']
create_layers_functions_dict = {'CONV':create_conv_layer, 'POOLMAX':create_pool_max_layer, 'POOLAVG':create_pool_avg_layer, 'DENSE':create_dense_layer}

ast = 50*'+'
SIZE_GENLIST = 10
NUM_FIXED_LAYERS = 5 #NUMBER OF FIXED LAYERS (INPUT, CONV AND FLATTEN DENSE DENSE) ===========================================================================================
INPUT_SIZE = 32
BATCH_SIZE = 64

#Possible values for different hyperparameters =====================
#                     0    1    2    3   4    5
MINMAX_LAYERS =      [3,   20]
CONV_KERNEL_LIST =   [3,   5]
POOL_KERN_LIST =     [2,   3]
NUM_FILTERS_LIST =   [32, 64,  128, 256]
DENSE_NEURONS_LIST = [10, 128, 256, 512]
ACTIVATION_FUNCTIONS_LIST = ['relu', 'sigmoid', 'tanh', 'softmax']
#==================================================================

#Set what layers in the gen_list may undergo changes.===========================================================
k = SIZE_GENLIST-NUM_FIXED_LAYERS #Number of layers between INP, CONV and FLATTEN, DENSE, DENSE
#All layers can change parameters, except the first one, FLATTEN and the last DENSE layer

k1 = k + 2 #All layers inbetween, plus the first CONV, plus the one before the last DENSE layer
MUTABLE_LCHANGEPARAM_INDEXES = list(range(1, k1)) + [k1 + 1]
 
#All layers can change type, except the first one, the CONV the FLATTEN and the two last DENSE layers
k2 = SIZE_GENLIST-NUM_FIXED_LAYERS #Number of layers between INP, CONV and FLATTEN, DENSE, DENSE
MUTABLE_LCHANGETYPE_INDEXES  = list(range(2, 2 + k))
#INDEXES FOR CROSSOVER
SPC_INDEXES = MUTABLE_LCHANGETYPE_INDEXES
#==============================================================================================================

#Set the minimum and maximum INDEX for the hyperparameters. This is used for mutation (indexes are added) =====
CONV_MINKERNEL_IND = 0
CONV_MINFILTER_IND = 0
POOL_MINKERNEL_IND = 0
DENSE_MINNEURONS_IND = 1 #Because the first one is 10, and it's used for the last DENSE layer.
ACTIVATION_MIN_IND = 0
CONV_MAXKERNEL_IND = len(CONV_KERNEL_LIST)-1
CONV_MAXFILTER_IND = len(NUM_FILTERS_LIST)-1
POOL_MAXKERNEL_IND = len(POOL_KERN_LIST)-1
DENSE_MAXNEURONS_IND = len(DENSE_NEURONS_LIST)-1
ACTIVATION_MAX_IND = len(ACTIVATION_FUNCTIONS_LIST)-1
#=============================================================================================================


#Each parameter is encoded as an integer for the genetic operators ===========================================
#NUM_FILTERS = {0:64, 1:128, 2:256, 3:512}
CONV_KERNELS = list_to_dictionary(CONV_KERNEL_LIST)
POOL_KERNELS = list_to_dictionary(POOL_KERN_LIST)
#NUM_FILTERS = {0:64, 1:128, 2:256, 3:512}
NUM_FILTERS = list_to_dictionary (NUM_FILTERS_LIST)
DENSE_NEURONS = list_to_dictionary(DENSE_NEURONS_LIST)
ACTIVATION_FUNCTIONS = list_to_dictionary(ACTIVATION_FUNCTIONS_LIST)
LAYERS_TYPES = list_to_dictionary(layer_types_list)
#==============================================================================================================

#This is used for the mutate_layer_parameters method
LAYER_DICTS_ASSOCIATION = {'CONV':CONV_KERNELS, 'POOLMAX':POOL_KERNELS, 'POOLAVG':POOL_KERNELS, 'DENSE':DENSE_NEURONS}

ARCH_NAMES_LIST = generate_letter_list()

# Get current date and hour
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M")
# Create filename
architecture_csv_filename  = f"archs_{formatted_time}.csv"

print(f'{MUTABLE_LCHANGETYPE_INDEXES = }')
print(f'{MUTABLE_LCHANGEPARAM_INDEXES = }')
