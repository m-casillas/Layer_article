from globalsENAS import *

#search_strategies_list = ['RANDOM', 'GA']
search_strategies_list = ['GA', 'RANDOM']
mutation_types_list = ['L_MODIFY_PARAMS', 'L_CHANGE_TYPE']
crossover_types_list = ['SPC', 'TPC', 'UC']

#search_strategies_list = ['GA']
#mutation_types_list = ['L_MODIFY_PARAMS', 'L_CHANGE_TYPE']
crossover_types_list = ['SPC', 'UC']


search_strategy_dict = list_to_dictionary(search_strategies_list)
mutation_type_dict = list_to_dictionary(mutation_types_list)
crossover_type_dict = list_to_dictionary(crossover_types_list)

#search_idx = 0
#mutation_idx = 1
#crossover_idx = 1

#search_strategy = search_strategy_dict[search_idx]
#mutation_type = mutation_type_dict[mutation_idx]
#crossover_type = crossover_type_dict[crossover_idx]

if search_strategies_list == ['RANDOM']:
    mutation_type = 'NONE'
    crossover_type = 'NONE'
    mutation_types_list = ['NONE']
    crossover_types_list = ['NONE']

SEED = 1
local_seed = SEED

DATASET_PART = 10 #Divide the dataset
MAIN_NPOP = 4
MUT_PROB = 1/(SIZE_GENLIST-NUM_FIXED_LAYERS) #SIZE_GENLIST is the number of layers that may mutate
EPOCHS = 2 #Number of epochs for training one architecture
GENERATIONS = 2
EXECUTIONS = 2 #1 execution is after #GENERATIONS

isRANDOM = 0
#Add to TOTAL_ARCH the number of architectures trained by RANDOM search
if 'RANDOM' in search_strategies_list:
    isRANDOM = 1
TOTAL_ARCH = 3*MAIN_NPOP*GENERATIONS*EXECUTIONS*len(mutation_types_list)*len(crossover_types_list) + MAIN_NPOP*GENERATIONS*EXECUTIONS*isRANDOM

TRAIN = True
PLOT = False
REPORT_ARCH = True #Save arch info in CSV

print(f'TOTAL ARCHITECTURES: {TOTAL_ARCH}')