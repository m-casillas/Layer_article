from globalsENAS import *

search_strategies_list = ['RANDOM', 'GA']
mutation_types_list = ['L_MODIFY_PARAMS', 'L_CHANGE_TYPE', 'NONE']
crossover_types_list = ['SPC', 'TPC']#, 'UC']


search_strategy_dict = list_to_dictionary(search_strategies_list)
mutation_type_dict = list_to_dictionary(mutation_types_list)
crossover_type_dict = list_to_dictionary(crossover_types_list)

search_idx = 1
mutation_idx = 1
#crossover_idx = 1

search_strategy = search_strategy_dict[search_idx]
mutation_type = mutation_type_dict[mutation_idx]
#crossover_type = crossover_type_dict[crossover_idx]

if search_strategy == 'RANDOM':
    mutation_type = 'NONE'

DATASET_PART = 1 #Divide the dataset
MAIN_NPOP = 8
MUT_PROB = 1/(SIZE_GENLIST-3) #SIZE_GENLIST is the number of layers that may mutate
EPOCHS = 8 #Number of epochs for training one architecture
GENERATIONS = 3
EXECUTIONS = 1 #1 execution is after #GENERATIONS

TRAIN = True
PLOT = False
REPORT_ARCH = True #Save arch info in CSV