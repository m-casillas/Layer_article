from globalsENAS import *

search_strategy_dict = {0:'RANDOM', 1:'GA'}
mutation_type_dict = {0:'L_MODIFY_PARAMS', 1:'L_CHANGE_TYPE', 2:'NONE'}
search_idx = 1
mutation_idx = 1

search_strategy = search_strategy_dict[search_idx]
mutation_type = mutation_type_dict[mutation_idx]
if search_strategy == 'RANDOM':
    mutation_type = 'NONE'
DATASET_PART = 10 #Divide the dataset
MAIN_NPOP = 10
MUT_PROB = 1/5 #5 is the number of layers
EPOCHS = 10 #Number of epochs for training one architecture
GENERATIONS = 3
EXECUTIONS = 5 #1 execution is after #GENERATIONS

TRAIN = True
RUN_ENAS = False #Run ENAS or print the grouped bar and blox plots