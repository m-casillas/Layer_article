from globalsENAS import *

search_strategies_list = ['RANDOM', 'GA']
mutation_types_list = ['L_MODIFY_PARAMS', 'L_CHANGE_TYPE', 'NONE']
crossover_types_list = ['SPC', 'TPC']


search_strategy_dict = list_to_dictionary(search_strategies_list)
mutation_type_dict = list_to_dictionary(mutation_types_list)
crossover_type_dict = list_to_dictionary(crossover_types_list)

search_idx = 1
mutation_idx = 0
crossover_idx = 0

search_strategy = search_strategy_dict[search_idx]
mutation_type = mutation_type_dict[mutation_idx]
crossover_type = crossover_type_dict[crossover_idx]

if search_strategy == 'RANDOM':
    mutation_type = 'NONE'
    
DATASET_PART = 10 #Divide the dataset
MAIN_NPOP = 20
MUT_PROB = 0.8#1/5 #5 is the number of layers
EPOCHS = 2 #Number of epochs for training one architecture
GENERATIONS = 5
EXECUTIONS = 10 #1 execution is after #GENERATIONS

TRAIN = False
RUN_ENAS = False #Run ENAS or print the grouped bar and blox plots
PLOT = False