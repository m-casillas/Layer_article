import sklearn
print(sklearn.__version__)

import random
def set_FLAGS(Surrogate, Train, Simulate):
        return Surrogate, Train, Simulate

def set_strategy_flags(SO_NORMAL, SO_HHSE, NSGA2_NORMAL, NSGA2_HHSE):
    if SO_NORMAL:
        HHSE, NSGA2 = False, False
    elif SO_HHSE:
        HHSE, NSGA2 = True, False
    elif NSGA2_NORMAL:
        HHSE, NSGA2 = False, True
    elif NSGA2_HHSE:
        HHSE, NSGA2 = True, True
    else:
        HHSE, NSGA2 = None, None
    return HHSE, NSGA2

def list_to_dictionary(list1):
     return {i:value for i,value in enumerate(list1)}

def set_mutation_types_list(REPRESENTATION_TYPE, ENCODING_TYPE):
    if REPRESENTATION_TYPE == 'L' and ENCODING_TYPE == 'IV':
        mutation_types_list = ['MPAR', 'MTYP', 'NONE'] #SWAP

    elif REPRESENTATION_TYPE == 'B' and ENCODING_TYPE == 'IV':
        mutation_types_list = ['MPAR', 'MSWAP', 'NONE']

    elif REPRESENTATION_TYPE == 'L' and ENCODING_TYPE == 'BV':
        mutation_types_list = ['MBFLIP']#, 'MBUNIF']
    else:
        mutation_types_list = []
    return mutation_types_list

#------------------------------------------------------------- EXPERIMENT SETTINGS ------------------------------------------------------------#

experiment_folder = "BLOCKS_70EPOCHS"#"experiment02_CIFAR100"#'test_experiment'# "experiment01_CIFAR10" # experiment02_CIFAR100"
REPRESENTATION_TYPE = 'B'  #L: layer, B: block
ENCODING_TYPE = 'IV' #IV: integer vector, BV: binary vector

search_strategies_list = ['GA', 'RANDOM']
crossover_types_list = ['SPC', 'TPC', 'UC', 'NONE']
mutation_types_list = set_mutation_types_list(REPRESENTATION_TYPE, ENCODING_TYPE)

#search_strategies_list = ['GA']
#crossover_types_list = ['UC']
#mutation_types_list = ['MSWAP']

crossoverList_hhse = ['SPC', 'TPC', 'UC']
mutationList_hhse = ['MPAR', 'MSWAP']
#------------------------------------------------------------- SEARCH STRATEGY ------------------------------------------------------------#
#Single Objective Normal
#Single Objective HHSE
#Multi Objective Normal (with NSGA2)
#Multi Objective HHSE (with NSGA2)
SO_NORMAL, SO_HHSE, NSGA2_NORMAL, NSGA2_HHSE = (0, 0, 1, 0)
HHSE, NSGA2 = set_strategy_flags(SO_NORMAL, SO_HHSE, NSGA2_NORMAL, NSGA2_HHSE)
HHSE_Tree, HHSE_RANDOM, HHSE_GREEDY = (1,0,0) if HHSE else (0,0,0)     #<--------- Change this one
PLOT_PARETO = False
#------------------------------------------------------------- \SEARCH STRATEGY ------------------------------------------------------------#

INITIAL_SEED = 666
RANDOMIZE_SEED = False

DATASET_PART = 1 #Percentage of the DATASET as decimal.
MAIN_NPOP = 100
EPOCHS = 1 #Number of epochs for training one architecture
GENERATIONS = 30
EXECUTIONS = 30#1 execution is after #GENERATIONS
SEED_LIST = [i for i in range(INITIAL_SEED, INITIAL_SEED + EXECUTIONS)] if RANDOMIZE_SEED == False else random.sample(range(100000), EXECUTIONS)
REGRESSOR_TYPE = 10

SURROGATE, TRAIN, SIMULATE = set_FLAGS(1, 0, 0)


#------------------------------------------------------------- REPORTING ------------------------------------------------------------#
REPORT_ARCH = True #Save arch info in CSV
REPORT_ONLY_BEST = True #Save BEST archs per generation. False saves all archs.
REPORT_ALL_COLUMNS = False #Save full arch info in CSV
RANDOM_NAMES = True #Random names for architecture instead of concatenating parents names.
REPORT_BATCH_SIZE = GENERATIONS #How many archs are saved in the CSV file at once.
#------------------------------------------------------------- REPORTING ------------------------------------------------------------#

PRINT1, PRINT2, PRINT3 = set_FLAGS(1, 0, 0) #Print levels


SORT_IDX = 0
SORT_ARCHS_list = ['SUPERIOR', 'MIDDLE', 'INFERIOR'] #Used for finding best, median or worst archs in a generation
SORT_ARCHS_DICT = list_to_dictionary(SORT_ARCHS_list)
SORT_ARCHS = SORT_ARCHS_DICT[SORT_IDX] 
DATASET_LIST = ['CIFAR10', 'CIFAR100']
DATASET_NUMDENSE_DICT = {'CIFAR10': 10, 'CIFAR100': 100}

plot_archcolumns = ['Accuracy', 'FLOPs', 'Num_Params']#, 'Top1', 'Top5']
#plot_archcolumns = ['Accuracy', 'cm_precision_macro', 'cm_recall_macro', 'cm_f1_macro', 'FLOPs', 'Num_Params']#, 'Top1', 'Top5']
plot_GAcolumns = ['HD_P1', 'HD_P2', 'HD_BM', 'HD_PB', 'Succ_Crossover_ratio', 'Succ_Mutation_ratio', 'NFHT', 'HV']
plot_convergency_columns = ['Accuracy','HD_PB','HV', 'Succ_Crossover_ratio', 'Succ_Mutation_ratio']
remove_report_columns = ['Epochs', 'Accuracy_history', 'Top1', 'Top5', 'Loss', 'SizeMB', 'Loss_history', 'CPU_Sec', 'P1_idx', 'P2_idx', 'P1', 'P2', 'Search_strategy'] #Discard these columns in report
                              #'Succ_Crossover', 'Succ_Mutation', 'Total_Crossovers', 'Total_Mutations']

#remove_status_columns = ['Generation', 'best_accuracy', 'mean_accuracy', 'mean_dP1', 'mean_dP2', 'mean_dBM', 'dPB',' HV', 'dHV', 'succ_cross', 'succ_mut']                
remove_status_columns = ['Execution', 'Crossover_type', 'Mutation_type', 'Chosen_operator', 'Selected_by', 'GOs_history']

#plot_GA_RANDOM = ['NFHT']
#plot_GA_NONECROSS = ['HD_BM', 'Succ_Mutation_ratio', 'NFHT']
#plot_GA_NONEMUT = plot_GA_RANDOM#['HD_P1', 'HD_P2', 'Succ_Crossover_ratio', 'NFHT']
