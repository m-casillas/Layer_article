from globalsENAS import *
os.system("cls")

search_strategies_list = ['GA', 'RANDOM']
mutation_types_list = ['MPARAMS', 'MTYPE', 'NONE']
crossover_types_list = ['SPC', 'TPC', 'UC', 'NONE']

search_strategies_list = ['RANDOM']
#mutation_types_list = ['MPARAMS']
#crossover_types_list = ['NONE']

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

RANDOMIZE_SEED = False
SEED = 42

DATASET_PART = 1/2 #Divide the dataset
MAIN_NPOP = 10
MUT_PROB = 1/(SIZE_GENLIST-NUM_FIXED_LAYERS) #SIZE_GENLIST is the number of layers that may mutate
EPOCHS = 10 #Number of epochs for training one architecture
GENERATIONS = 10
EXECUTIONS = 10 #1 execution is after #GENERATIONS

isRANDOM = 0
isGA = 1
#Add to TOTAL_ARCH the number of architectures trained by RANDOM search. isRandom and isGA are factors
if 'RANDOM' in search_strategies_list:
    isRANDOM = 1
    if len(search_strategies_list) == 1: #Only RANDOM
        isGA = 0
num_mutations = len(mutation_types_list)
num_crossovers = len(crossover_types_list)
print(f'{MUT_PROB = }, {EPOCHS = }, {MAIN_NPOP = }, {GENERATIONS = }, {EXECUTIONS = }, {isRANDOM = }, {isGA = }')
TOTAL_ARCH = (MAIN_NPOP*GENERATIONS + 2*MAIN_NPOP*(GENERATIONS-1))*EXECUTIONS*num_mutations*num_crossovers*isGA + MAIN_NPOP*GENERATIONS*EXECUTIONS*isRANDOM

SURROGATE = True
TRAIN = False
SIMULATE = False

PLOT = False
REPORT_ARCH = True #Save arch info in CSV

print(f'TOTAL ARCHITECTURES: {TOTAL_ARCH}')