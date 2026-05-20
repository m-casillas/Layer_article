import os, platform

def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_running_in_colab():
    print("Running in Google Colab.")
    path =  '/content/drive/MyDrive/DCC/Research/Layer_article'
elif platform.system() == 'Windows':
    print("Running in Windows.")
    path =  os.getcwd()
else:
    print('Running in NVIDIA-CEM')
    #print("Running in Azure VM.")
    #path = '/home/Super-IR/Thesis_code/Tesis_code/Layer_article'
    path = os.getcwd()




from colorama import Fore, Back, Style, init
init(autoreset=True)
import config_tecnas
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import psutil
import shutil
import math
import sys
import subprocess
import ast
from abc import ABC, abstractmethod

import io
import itertools
from itertools import combinations
import platform
import contextlib
import logging
from datetime import datetime

#from keras.src.callbacks import EarlyStopping
#from keras.src.layers import Input
#from keras.src import datasets, layers, models
#from keras.src import backend as K
#from tensorflow.keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split
from pympler import asizeof
import tracemalloc

def is_HHSE(filename):
    return True if 'HHSE' in filename else False


def calculate_meanMetric_arch(archList, metric_str):
    #Calculate mean of a metric for a list of architectures
    return  sum(getattr(arch,metric_str) for arch in archList) / len(archList)


def remove_duplicates(df, column_name='Integer_encoding'):
    if column_name not in df.columns:
        raise ValueError(f"The DataFrame must contain a '{column_name}' column.")
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=[column_name])
    removed_count = initial_count - len(df_cleaned)
    print(Fore.YELLOW + f"Removed {removed_count} duplicate entries based on {column_name}" + Style.RESET_ALL)
    return df_cleaned

def dprint(verbose = True, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def remove_keys(original_dict, listofKeys_to_remove):
    #Returns a new dictionary without the specified keys.
    return {k: v for k, v in original_dict.items() if k not in listofKeys_to_remove}


def bitflip(bit_string, idx):
        #Flip the bit at position idx
        if bit_string[idx] == '0':
            bit_string = bit_string[:idx] + '1' + bit_string[idx+1:]
        else:
            bit_string = bit_string[:idx] + '0' + bit_string[idx+1:]
        return bit_string
        


def gray_to_int(gray_str):
    #Convert a Gray code bitstring back to its integer value.
    gray = int(gray_str, 2)
    mask = gray
    while mask > 0:
        mask >>= 1
        gray ^= mask
    return gray

def int_to_gray(i, N):
    #Convert an integer n to its Gray code representation as a bitstring.
    #Uses the minimum number of bits needed to represent N.
    bits = N.bit_length()
    gray = i ^ (i >> 1)  # compute Gray code
    return format(gray, f'0{bits}b')


def index_lists(lists):
    """
    Given a list of lists, return the list of index lists
    showing where each sublist lives in the concatenated list.
    For example, indexes for all_convs, indexes for all_pools, etc.
    """
    indices = []
    start = 0
    for sublist in lists:
        end = start + len(sublist)
        indices.append(list(range(start, end)))
        start = end
    return indices

def filter_csv(df_folder_path = '', df = None, generation = 'ALL', arch_status = 'BEST', rank = 'ALL', nparts = 1) :
    def filter_df(df):
        
        conditions_list = []
        rank_dict = {'ALL':'A','HIGHEST':'1', 'MIDDLE':'M'}
        if rank != 'ALL': #Keep architectures with ranking rank. Only one architecture per combination of crossover and mutation
            print(f'Generation: ALL, Arch_status: BEST, Rank: {rank}, Nparts: {nparts}')
            if rank == 'HIGHEST':
                if 'HHSE' in df.columns and df['HHSE'].sum() > 0: #If this is a HHSE experiment, get the biggest of all rank 1 architectures
                    df = df[(df['arch_status'] == 'BEST')]
                    df = df.sort_values('Accuracy', ascending=False).iloc[[0]]
                else:
                    df = df[(df["Ranking"] == 1) & (df['arch_status'] == 'BEST')]
                    df = df.drop_duplicates(subset=["Crossover_type", "Mutation_type"], keep="first")

                

            elif rank == 'MIDDLE':
                df = df[(df['arch_status'] == 'BEST')]
                medians = df.groupby(["Crossover_type", "Mutation_type"])["Ranking"].median().reset_index()
                medians.rename(columns={"Ranking":"median_R"}, inplace=True)
                df_with_median = df.merge(medians, on=["Crossover_type", "Mutation_type"])
                df_with_median["diff"] = (df_with_median["Ranking"] - df_with_median["median_R"]).abs()
                result = df_with_median.loc[df_with_median.groupby(["Crossover_type", "Mutation_type"])["diff"].idxmin()]
                df = result.drop(columns=["median_R", "diff"]).reset_index(drop=True)
        else:
            print(f'Generation: {generation}, Arch_status: {arch_status}, Rank: {rank}, Nparts: {nparts}')
            if arch_status != 'ALL':
                if arch_status == 'BLANK':
                    conditions_list.append(df['arch_status'].isna())
                else:
                    conditions_list.append(df['arch_status'] == arch_status)
            
            if generation == 'LAST':
                conditions_list.append(df['Generation'] == df['Generation'].max())

            #Apply all conditions
            if len(conditions_list) != 0:
                mask = conditions_list[0]
                for c in conditions_list[1:]:
                    mask = mask & c
                df = df[mask]
        print(f'Filtered to {len(df)} rows')
        if nparts > 1:
            split_indices = np.array_split(df, nparts)
            for i, part in enumerate(split_indices, start=1):
                print(f"Part {i}: {len(part)} rows")
            for i, part in enumerate(split_indices, start=1):
                filename = f"_rank{rank_dict[rank]}_part{i}.csv"
                full_path = df_full_path[:-4] + filename
                part.to_csv(full_path, index=False)
                print(f"Saved {full_path}")
        else:
            filename = f"_rank{rank_dict[rank]}_{arch_status}.csv"
            full_path = df_full_path[:-4] + filename
            df.to_csv(full_path, index=False)
            print(f"Saved {full_path}")
    #Created specially for filtering by rank, for the CM
    
    if df_folder_path == '':
        filter_df(df)
    else:
        for filename in os.listdir(df_folder_path):
            if filename.endswith(".csv"):
                if 'filtered' in filename:
                    print(f'Skipping {filename} because it is already filtered.')
                    continue
                df_full_path = os.path.join(df_folder_path, filename)
                print(f'Filtering {df_full_path}')
                df = pd.read_csv(df_full_path)
                filter_df(df)


def rank_archs(df_full_path):
    #Assigns a rank for BEST archs, according to the accuracy, per Combination.
    df = pd.read_csv(df_full_path)
    print(f'Ranking {df_full_path}')
    df["Ranking"] = df[df['arch_status'] == 'BEST'].groupby(['Crossover_type', 'Mutation_type'])["Accuracy"].rank(method="dense", ascending=False).astype("Int64")
    df.to_csv(df_full_path, index = False)
    print('Finished')


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")


def list_to_dictionary(list1):
     return {i:value for i,value in enumerate(list1)}
     
def generate_letter_list():
    #Generate letters from A to ZZZZ, for the architecture idx. That dependes on the size in range.
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
    dummy_input = tf.random.normal((1,) + model.input_shape[1:])
    concrete_func = tf.function(model).get_concrete_function(dummy_input)

    options = ProfileOptionBuilder.float_operation()
    prof = profile(concrete_func.graph, options=options)
    flops = prof.total_float_ops // 2
    return flops

def calculate_model_params(model):
    total_params = 0
    total_bytes = 0
    trainable_params = 0
    non_trainable_params = 0
    # ---- Trainable weights ----
    for w in model.trainable_weights:
        n = np.prod(w.shape)
        dtype_size = tf.as_dtype(w.dtype).size  # SAFE conversion
        param_bytes = n * dtype_size
        trainable_params += n
        total_params += n
        total_bytes += param_bytes

    # ---- Non-trainable weights ----
    for w in model.non_trainable_weights:
        n = np.prod(w.shape)
        dtype_size = tf.as_dtype(w.dtype).size
        param_bytes = n * dtype_size
        non_trainable_params += n
        total_params += n
        total_bytes += param_bytes

    size_mb = total_bytes / (1024 ** 2)
    return total_params, size_mb

def create_pool_max_layer(ks = None):
        #A Pool layer has: kernel size (ks)... more to come
        #If ks is None, it is randomly created
        if ks == None:
            ks = random.choice(list(Globals.POOL_KERNELS.values()))
        return {'POOLMAX':[-1, ks]} #-1 is neccesary to keep pool layers the same size as conv layers. It will never mutate

def create_pool_avg_layer(ks = None):
        #A Pool layer has: kernel size (ks)... more to come
        #If ks is None, it is randomly created
        if ks == None:
            ks = random.choice(list(Globals.POOL_KERNELS.values()))
        return {'POOLAVG':[-1, ks]} #-1 is neccesary to keep pool layers the same size as conv layers. It will never mutate

def create_globalAVG_layer():
        return {'GLOBAL_AVG':None}

def create_conv_layer(nf = None, ks = None):
        #A Conv layer has: Number of filters (nf) kernel size (ks)... more to come
        #If nf and ks are None, it is randomly created
        if nf == None and ks == None:
            #{'CONV':[NUM_FILTERS[np.random.randint(0, len(NUM_FILTERS))],CONV_KERNELS[np.random.randint(0, len(CONV_KERNELS))]]}, This is for MUTATION
            nf = random.choice(list(Globals.NUM_FILTERS.values())) 
            ks = random.choice(list(Globals.CONV_KERNELS.values()))
        return {'CONV':[nf, ks]}

def create_dense_layer(nn = None, act = None, last_dense = False):
        #A Dense layer has: Number of neurons (nn) and activation function (act)... more to come
        #If nn and act are None, it is randomly created
        type_dense = 'DENSE'
        if nn == None and act == None and last_dense == False:
            nn = random.choice(Globals.DENSE_NEURONS_LIST) #Ignore the first number (10), that's for the last layer
            act = 'relu'
            type_dense = 'DENSE'
        if last_dense == True:
            nn = Globals.NUM_CLASSES
            act = 'softmax'
            type_dense = 'LAST_DENSE'
        return {type_dense:[nn, act]}

def create_last_name(column_list):
    last_name = ''
    for c in column_list:
            last_name = last_name + c + '_'
    return last_name

def determine_label_filename2(filename):
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

def get_key_from_dict(dictio):
    #Returns the first key of a dictionary.
    return list(dictio.keys())[0]

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
        arch_obj.dBM = hamming_distance(arch_obj.integer_encoding, arch_obj.before_mutation.integer_encoding)/arch_obj.integer_size
    else:
        arch_obj.dP1 = hamming_distance(arch_obj.integer_encoding, arch_obj.parent1.integer_encoding)/arch_obj.integer_size
        arch_obj.dP2 = hamming_distance(arch_obj.integer_encoding, arch_obj.parent2.integer_encoding)/arch_obj.integer_size

         
def is_None_or_empty(object):
    #Check if an object is None or empty
    if object == None or object == []:
        return True
    return False

def check_within_bounds(number, LB, UB):
    #Used for keeping indexes that are going to change within the bounds. If it gets out of bounds, it wraps around.
    if number > UB:
        number = LB
    elif number < LB:
        number = UB
    return number

def random_choice_except(lst, elem):
    #Returns a random string from lst, excluding elem
    #Used for mutation by selecting an operation, excluding the current one.
    #For example, if op_type is CONV, it returns another layer type, like POOLMAX or POOLAVG
    filtered_list = [item for item in lst if item != elem]
    return random.choice(filtered_list) if filtered_list else None



class BLOCKS_CONSTANTS:
    SIZE_GENLIST = 12
    SPC_INDEXES = list(range(2,SIZE_GENLIST-2)) #All blocks in between. Skip the INPat the beginning and GLOBAL and DENSE at the end.
    MUTABLE_BCHANGEPARAM_INDEXES = SPC_INDEXES
    MUTABLE_BSWAP_INDEXES = SPC_INDEXES

class LAYERS_CONSTANTS:
    #        0                 1                 2                 3                 4                           5                 6                 7                 8                        9        
    #[{'INP': 32}, {'CONV': [32, 3]}, {'POOLMAX': [-1, 3]}, {'CONV': [256, 3]}, {'POOLMAX': [-1, 3]}, {'CONV': [32, 5]}, {'CONV': [128, 5]}, {'FLATTEN': None}, {'DENSE': [64, 'relu']}, {'DENSE': [10, 'softmax']}]
    #[{'INP': 32}, {'CONV': [64, 3]}, {'POOLMAX': [-1, 5]}, {'CONV': [128, 3]}, {'POOLMAX': [-1, 5]}, {'CONV': [16, 5]}, {'CONV': [128, 5]}, {'FLATTEN': None}, {'DENSE': [64, 'relu']}, {'DENSE': [10, 'softmax']}]
    SIZE_GENLIST = 10
    NUM_FIXED_LAYERS = 5 #NUMBER OF FIXED LAYERS (INPUT, CONV AND FLATTEN DENSE DENSE) ===========================================================================================
    NUM_MUTABLE_LAYERS = NUM_FIXED_LAYERS + 2 # (add the first CONV, last POOL and before last DENSE)
    #Set what layers in the gen_list may undergo changes.===========================================================
    k = SIZE_GENLIST-NUM_FIXED_LAYERS #Number of layers between INP, CONV and FLATTEN, DENSE, DENSE
    #All layers can change parameters, except the first one, FLATTEN and the last DENSE layer

    k1 = k + 2 #All layers inbetween, plus the first CONV, plus the one before the last DENSE layer
    MUTABLE_LCHANGEPARAM_INDEXES = list(range(1, k1)) + [k1 + 1]
    
    #All layers can change type, except the first one, the CONV the FLATTEN and the two last DENSE layers
    k2 = SIZE_GENLIST-NUM_FIXED_LAYERS #Number of layers between INP, CONV and FLATTEN, DENSE, DENSE
    MUTABLE_LCHANGETYPE_INDEXES  = list(range(2, 2 + k))

    MUTABLE_LBITFLIP_INDEXES = MUTABLE_LCHANGETYPE_INDEXES
    #INDEXES FOR CROSSOVER
    SPC_INDEXES = list(range(1, 2 + k))
    #==============================================================================================================

class ConfigClass:
    

    def define_random(search_strategies_list, mutation_types_list, crossover_types_list):
        #Defines the random strategy, where mut and cross are NONE NONE
        if 'NONE' in mutation_types_list and 'NONE' in crossover_types_list and 'RANDOM' not in search_strategies_list:
            search_strategies_list.append('RANDOM')

        if 'NONE' in mutation_types_list and 'NONE' in crossover_types_list and (len(mutation_types_list) + len(crossover_types_list) == 2):
            search_strategies_list = ['RANDOM']

        if 'RANDOM' in search_strategies_list:
            #If it's only RANDOM, remove other possible search strategies
            if search_strategies_list == ['RANDOM']:
                mutation_types_list = ['NONE']
                crossover_types_list = ['NONE']
            #If RANDOM is explicitly defined, check that NONE are in the cross and mut lists. If not, add them.
            if 'NONE' not in mutation_types_list:
                mutation_types_list.append('NONE')
            if 'NONE' not in crossover_types_list:
                crossover_types_list.append('NONE')
        return search_strategies_list, mutation_types_list, crossover_types_list

    @classmethod
    def number_of_archs(cls, search_strategies_list, mutation_types_list, crossover_types_list):
        NUM_ARCH_RAND = NUM_HALF_GA = NUM_COMPLETE_GA = 0
        num_mutations_types = num_crossovers_types = 1
        NO_NONE_mutation_types_list  = [item for item in mutation_types_list if item != 'NONE']
        NO_NONE_crossover_types_list = [item for item in crossover_types_list if item != 'NONE']
        NO_NONE_num_mutations_types  = len(NO_NONE_mutation_types_list)
        NO_NONE_num_crossovers_types = len(NO_NONE_crossover_types_list)
        if 'RANDOM' in search_strategies_list:
            NUM_ARCH_RAND = cls.EXECUTIONS*cls.GENERATIONS*cls.MAIN_NPOP
            if search_strategies_list == ['RANDOM']: #If it's only RANDOM, return its cardinality
                return NUM_ARCH_RAND
            else: #This means both CROSS-NONE and NONE-MUT are included. Substact the overlapping number of random archs.
                NUM_CROSS_NONE = NO_NONE_num_crossovers_types*cls.EXECUTIONS*(2*cls.MAIN_NPOP + cls.MAIN_NPOP*(cls.GENERATIONS-1))
                NUM_MUT_NONE   = NO_NONE_num_mutations_types*cls.EXECUTIONS*(2*cls.MAIN_NPOP + cls.MAIN_NPOP*(cls.GENERATIONS-1))
                NUM_HALF_GA = NUM_CROSS_NONE + NUM_MUT_NONE
                print(f'{NUM_ARCH_RAND = }, {NUM_CROSS_NONE = }, {NUM_MUT_NONE = }, {NUM_HALF_GA = }')
        
        else:
            if ('NONE' in mutation_types_list) and ('NONE' not in crossover_types_list):
                num_crossovers_types = len(crossover_types_list)
                NUM_HALF_GA = num_crossovers_types*cls.EXECUTIONS*(2*cls.MAIN_NPOP + cls.MAIN_NPOP*(cls.GENERATIONS-1))
            if ('NONE' in crossover_types_list) and ('NONE' not in mutation_types_list):
                num_mutations_types = len(mutation_types_list)
                NUM_HALF_GA = num_mutations_types*cls.EXECUTIONS*(2*cls.MAIN_NPOP + cls.MAIN_NPOP*(cls.GENERATIONS-1))
        
        NUM_COMPLETE_GA = NO_NONE_num_mutations_types*NO_NONE_num_crossovers_types*cls.EXECUTIONS*(3*cls.MAIN_NPOP + 2*cls.MAIN_NPOP*(cls.GENERATIONS-1))
        TOTAL_ARCH = NUM_COMPLETE_GA + NUM_ARCH_RAND + NUM_HALF_GA
        return TOTAL_ARCH

    experiment_folder = config_tecnas.experiment_folder
    REPRESENTATION_TYPE = config_tecnas.REPRESENTATION_TYPE
    ENCODING_TYPE = config_tecnas.ENCODING_TYPE
    search_strategies_list = config_tecnas.search_strategies_list
    crossover_types_list = config_tecnas.crossover_types_list
    mutation_types_list = config_tecnas.mutation_types_list
    search_strategies_list, mutation_types_list, crossover_types_list = define_random(search_strategies_list, mutation_types_list, crossover_types_list) #Define RANDOM if needed

    search_strategy_dict = list_to_dictionary(search_strategies_list)
    mutation_type_dict = list_to_dictionary(mutation_types_list)
    crossover_type_dict = list_to_dictionary(crossover_types_list)

    SEED = config_tecnas.SEED_LIST[0]
    RANDOMIZE_SEED = config_tecnas.RANDOMIZE_SEED

    DATASET_PART = config_tecnas.DATASET_PART #Percentage of the DATASET
    MAIN_NPOP = config_tecnas.MAIN_NPOP
    MUT_PROB = 1/LAYERS_CONSTANTS.NUM_MUTABLE_LAYERS if REPRESENTATION_TYPE == 'L' else 1/BLOCKS_CONSTANTS.SIZE_GENLIST
    EPOCHS = config_tecnas.EPOCHS #Number of epochs for training one architecture
    GENERATIONS = config_tecnas.GENERATIONS
    EXECUTIONS = config_tecnas.EXECUTIONS #1 execution is after #GENERATIONS
    REGRESSOR_TYPE = config_tecnas.REGRESSOR_TYPE

    SURROGATE, TRAIN, SIMULATE = config_tecnas.SURROGATE, config_tecnas.TRAIN, config_tecnas.SIMULATE
    REPORT_ARCH = config_tecnas.REPORT_ARCH #Save arch info in CSV
    RANDOM_NAMES = config_tecnas.RANDOM_NAMES #Random names for architecture instead of concatenating parents names.

    SORT_ARCHS_list = config_tecnas.SORT_ARCHS_list
    SORT_ARCHS_DICT = config_tecnas.SORT_ARCHS_DICT
    SORT_ARCHS = config_tecnas.SORT_ARCHS
    DATASET_LIST = config_tecnas.DATASET_LIST
    DATASET_NUMDENSE_DICT = config_tecnas.DATASET_NUMDENSE_DICT
ConfigClass.DATASET = ConfigClass.DATASET_LIST[0] if ConfigClass.REPRESENTATION_TYPE == 'L' else ConfigClass.DATASET_LIST[1]
ConfigClass.TOTAL_ARCH = ConfigClass.number_of_archs(ConfigClass.search_strategies_list, ConfigClass.mutation_types_list, ConfigClass.crossover_types_list)


class ConfigBlocks:
    NBLOCKS = 8
    NCONV_PERBLOCK = 2

class Globals:
    ast_bar = 50*'+'
    INPUT_SIZE = 32
    BATCH_SIZE = 64
    NUM_CLASSES = 10 if ConfigClass.DATASET == 'CIFAR10' else 100

    #Possible values for different hyperparameters =====================
    #                     0    1    2    3   4    5
    MINMAX_LAYERS =      [3,   20]
    CONV_KERNEL_LIST =   [3,   5]
    POOL_KERN_LIST =     [2,   3]
    NUM_FILTERS_LIST =   [32, 64,  128, 256]
    DENSE_NEURONS_LIST = [128, 256, 512]
    ACTIVATION_FUNCTIONS_LIST = ['relu', 'sigmoid', 'tanh', 'softmax']
    #==================================================================

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
#                           0       1         2        3            4        5            6           7
    layer_types_list =    ['INP', 'CONV', 'POOLMAX', 'POOLAVG', 'FLATTEN', 'DENSE', 'GLOBAL_AVG', 'LAST_DENSE']
    type_mutable_layers = ['CONV','POOLMAX']#,'POOLAVG']
    create_layers_functions_dict = {'CONV':create_conv_layer, 'POOLMAX':create_pool_max_layer, 'POOLAVG':create_pool_avg_layer, 'DENSE':create_dense_layer, 'GLOBAL_AVG':create_globalAVG_layer}
    #NUM_FILTERS = {0:64, 1:128, 2:256, 3:512}
    CONV_KERNELS = list_to_dictionary(CONV_KERNEL_LIST)
    POOL_KERNELS = list_to_dictionary(POOL_KERN_LIST)
    #NUM_FILTERS = {0:64, 1:128, 2:256, 3:512}
    NUM_FILTERS = list_to_dictionary (NUM_FILTERS_LIST)
    DENSE_NEURONS = list_to_dictionary(DENSE_NEURONS_LIST)
    ACTIVATION_FUNCTIONS = list_to_dictionary(ACTIVATION_FUNCTIONS_LIST)
    LAYERS_TYPES = list_to_dictionary(layer_types_list)

    @classmethod
    def create_all_convs_pools_denses(cls):
        #Creates three lists: one with all possible CONVS, another with all possible POOLMAX and another with all possible DENSE
        convs = [{'CONV': [n, k]} for n in cls.NUM_FILTERS_LIST for k in Globals.CONV_KERNEL_LIST]
        pools = [{'POOLMAX': [-1, k]} for k in cls.POOL_KERN_LIST]
        dens = [{'DENSE': [n, 'relu']} for n in cls.DENSE_NEURONS_LIST]
        return convs, pools, dens

    
    

    


ConfigClass.IMPORT_TENSORFLOW = True if ConfigClass.TRAIN == True else False
if ConfigClass.IMPORT_TENSORFLOW or config_tecnas.HHSE == True:
    from import_tf import *
    print(f'{tf.__version__=}')
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("GPU Details:", tf.config.list_physical_devices('GPU'))

#Lists of dictionaries
Globals.all_convs, Globals.all_pools, Globals.all_denses = Globals.create_all_convs_pools_denses()
Globals.all_convs_pools = Globals.all_convs + Globals.all_pools
Globals.all_layers = Globals.all_convs + Globals.all_pools + Globals.all_denses
indexes_layers = index_lists([Globals.all_convs, Globals.all_pools, Globals.all_denses]) 
Globals.INDEXES_CONVS = indexes_layers[0] #0 to 7 in Globals.all_layers
Globals.INDEXES_POOLS = indexes_layers[1] #8 and 9
Globals.INDEXES_DENSES = indexes_layers[2] #10 to 12


#This is used for the mutate_layer_parameters method
LAYER_DICTS_ASSOCIATION = {'CONV':Globals.CONV_KERNELS, 'POOLMAX':Globals.POOL_KERNELS, 'POOLAVG':Globals.POOL_KERNELS, 'DENSE':Globals.DENSE_NEURONS}
ARCH_NAMES_LIST = generate_letter_list()

# Get current date and hour
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M")
# Create filename
architecture_filename_noCSV  = f"{formatted_time}"

# ========================================================= CONFIG ==========================================================================



print(f'TOTAL ARCHITECTURES: {ConfigClass.TOTAL_ARCH}')
print('END OF configENAS information')
print('------------------------------------------------------------------------------------')
'''
0: XGBRegressor
1: RandomForestRegressor
2: Ridge
3: SVR
4: MLPRegressor
5: LinearRegression(),
6: Lasso
7: ElasticNet
8: BayesianRidge(),
9: HuberRegressor(),
10: GradientBoostingRegressor
11: AdaBoostRegressor
12: ExtraTreesRegressor
13: DecisionTreeRegressor
14: KNeighborsRegressor
15: GaussianProcessRegressor
'''

