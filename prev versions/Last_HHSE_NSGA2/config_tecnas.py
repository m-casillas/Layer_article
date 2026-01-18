import random
def set_FLAGS(Surrogate, Train, Simulate):
        return Surrogate, Train, Simulate

def list_to_dictionary(list1):
     return {i:value for i,value in enumerate(list1)}

experiment_folder = "BLOCKS_70EPOCHS"#"experiment02_CIFAR100"#'test_experiment'# "experiment01_CIFAR10" # experiment02_CIFAR100"
REPRESENTATION_TYPE = 'B'  #L: layer, B: block
ENCODING_TYPE = 'IV' #IV: integer vector, BV: binary vector

search_strategies_list = ['GA', 'RANDOM']
crossover_types_list = ['SPC', 'TPC', 'UC', 'NONE']

if REPRESENTATION_TYPE == 'L' and ENCODING_TYPE == 'IV':
    mutation_types_list = ['MPAR', 'MTYP', 'NONE'] #SWAP

elif REPRESENTATION_TYPE == 'B' and ENCODING_TYPE == 'IV':
    mutation_types_list = ['MPAR', 'MSWAP', 'NONE']

elif REPRESENTATION_TYPE == 'L' and ENCODING_TYPE == 'BV':
    mutation_types_list = ['MBFLIP']#, 'MBUNIF']

search_strategies_list = ['GA']
crossover_types_list = ['UC']
mutation_types_list = ['MPAR']


crossoverList_hhse = ['SPC', 'TPC', 'UC', 'NONE']
mutationList_hhse = ['MPAR', 'MSWAP', 'NONE']

#------------------------------------------------------------- HHSE -------------------------------------------------------------#                  
all_objective_names = ['Accuracy', 'FLOPs', 'Num_params', 'DP1', 'DP2', 'DBM', 'Succ_Cross', 'Succ_Mut', 'mean_Accuracy', 'mean_FLOPs', 'mean_Num_params', 'mean_DBM', 'mean_DP1', 'mean_DP2']
objective_maxmin_names = {'Accuracy': 'MAX', 'FLOPs': 'MIN', 'Num_params': 'MIN'} #{'Accuracy': 'MAX', 'DBM': 'MAX'} #{'Accuracy': 'MAX', 'FLOPs': 'MIN', 'Num_params': 'MAX'} 
#ACC: select operator with the biggest accuracy
#CROWD: select by crowding distance
#SUCCMUT: select operator with the highest successful mutations
NSGA_II_SELECTION_CSVREPORT = {'Accuracy':'ACC', 'FLOPs':'FLOPs','Num_params':'NUMPARAMS','Succ_Mut':'SUCCMUT', 'Crowd':'CROWD', 'rand':'RAND'}
NSGA_II_SELECTION = ['Succ_Mut', True] #Selects the best (True)/worst (False) genetic operator from the pareto front according to the column. If 'crowd' is select, uses crowding distance. If 'rand ' is select, selects randomly from the pareto front.
HHSE = False #Hyper-heuristic selection ****************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------#

INITIAL_SEED = 1111
RANDOMIZE_SEED = True

DATASET_PART = 1 #Percentage of the DATASET as decimal.
MAIN_NPOP = 30
EPOCHS = 2 #Number of epochs for training one architecture
GENERATIONS = 20
EXECUTIONS = 100 #1 execution is after #GENERATIONS
SEED_LIST = [i for i in range(INITIAL_SEED, INITIAL_SEED + EXECUTIONS)] if RANDOMIZE_SEED == False else random.sample(range(100000), EXECUTIONS)
REGRESSOR_TYPE = 10

SURROGATE, TRAIN, SIMULATE = set_FLAGS(1, 0, 0)
REPORT_ARCH = True #Save arch info in CSV
FULL_REPORT = True #Save arch info in CSV
RANDOM_NAMES = True #Random names for architecture instead of concatenating parents names.

SORT_IDX = 2
SORT_ARCHS_list = ['SUPERIOR', 'MIDDLE', 'INFERIOR'] #Used for finding best, median or worst archs in a generation
SORT_ARCHS_DICT = list_to_dictionary(SORT_ARCHS_list)
SORT_ARCHS = SORT_ARCHS_DICT[SORT_IDX] 
DATASET_LIST = ['CIFAR10', 'CIFAR100']
DATASET_NUMDENSE_DICT = {'CIFAR10': 10, 'CIFAR100': 100}

plot_archcolumns = ['Accuracy', 'cm_precision_macro', 'cm_recall_macro', 'cm_f1_macro', 'FLOPs', 'Num_Params']#, 'Top1', 'Top5']
plot_GAcolumns = ['HD_P1', 'HD_P2', 'HD_BM', 'Succ_Crossover_ratio', 'Succ_Mutation_ratio', 'NFHT']
plot_GA_RANDOM = ['NFHT']
plot_GA_NONECROSS = plot_GA_RANDOM#['HD_BM', 'Succ_Mutation_ratio', 'NFHT']
plot_GA_NONEMUT = plot_GA_RANDOM#['HD_P1', 'HD_P2', 'Succ_Crossover_ratio', 'NFHT']
