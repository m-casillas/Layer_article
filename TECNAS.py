import tensorflow as tf
from colorama import Fore, Back, Style, init
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
#from globalsENAS import *
from configENAS import *
from Genotype import *
from ReportENAS import *
from PlotterENAS import *
from LayerRepresentation import *
from Crossover import *
from Mutator import *
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

"""# TECNAS Classs"""

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#genotype = Genotype('L', 'IV', gen_list)
#arch = Architecture('X', 0, 0, 0, 0, 1, genotype)

#TECNAS is a class that performs evoltuionary neural architecture search, using genetic operators
class TECNAS:
    def get_normalize_dataset(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()
        # Get total number of training samples
        total_train_samples = self.x_train.shape[0]
        if DATASET_PART == 1:
            # Use the entire dataset
            subset_x_train = self.x_train
            subset_y_train = self.y_train
        else:
            # Use only a portion of the dataset
            subset_size = total_train_samples // DATASET_PART
            subset_x_train = self.x_train[:subset_size]
            subset_y_train = self.y_train[:subset_size]

        # Normalize pixel values to be between 0 and 1
        self.x_train, self.x_test = subset_x_train / 255.0, self.x_test / 255.0
        self.y_train, self.y_test = subset_y_train, self.y_test

    def make_child(self, arch, parent1, parent2):
        ind = copy.deepcopy(arch)
        ind.isChild = True
        ind.isMutant = False
        ind.parent1 = parent1
        ind.parent2 = parent2
        ind.before_mutation = None
        ind.integer_encoding = ind.genList_to_integer_vector()
        ind.dP1 = hamming_distance(ind.integer_encoding, parent1.integer_encoding)
        ind.dP2 = hamming_distance(ind.integer_encoding, parent2.integer_encoding)
        ind.dBM = -1
        return ind

    def make_mutant(self, archM, archOriginal):
        ind = copy.deepcopy(archM)
        ind.isChild = True
        ind.isMutant = True
        ind.parent1 = None
        ind.parent2 = None
        ind.before_mutation = archOriginal
        ind.integer_encoding = ind.genList_to_integer_vector()
        ind.dP1 = -1
        ind.dP2 = -1
        ind.dBM = hamming_distance(ind.integer_encoding, archOriginal.integer_encoding)
        return ind

    def make_parent(self, arch):
        ind = copy.deepcopy(arch)
        ind.isChild = False
        ind.isMutant = False
        ind.parent1 = None
        ind.parent2 = None
        ind.before_mutation = None
        ind.dP1 = -1
        ind.dP2 = -1
        ind.dBM = -1
        return ind

    def validate_architecture(self, gen_list):
        #Determine if the architecture is valid before training. Check that there are no consecutive POOL layers
        #Returns a valid gen_list by changing the POOL to CONV
        print(f'Validating:  {gen_list}')
        if len(gen_list) != SIZE_GENLIST:
            print(f'FATAL ERROR. ARCHITECTURE SIZE CHANGED: {len(gen_list)}, should be {SIZE_GENLIST}')
            return None
        changed = False
        temp_gen_list = copy.deepcopy(gen_list)
        for i, layer_dict in enumerate(gen_list): 
            layer_type = list(layer_dict.keys())[0] #Get the layer type
            if layer_type in ['POOLMAX', 'POOLAVG']:
                next_layer = list(gen_list[i+1].keys())[0]
                if next_layer in ['POOLMAX', 'POOLAVG']:
                    #print(f'\nCHANGING {temp_gen_list[i]}')
        #            print('ALERT ALERT ALERT ALERT ALERT ALERT ALERT')
                    temp_gen_list[i] = create_conv_layer()
                    changed = True
        if changed == True:                 
            print(f'Changed to {temp_gen_list}\n')
        return temp_gen_list
    
    def change_layer_parameters_or_type(self, arch_obj_ind, mutation_type):
        #gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
        #Mutate each layer with probability MUT_PROB.
        #Save it in the genotype and update it.
        #Return the mutated architecture
        mutator_obj = Mutator()

        if mutation_type == 'L_MODIFY_PARAMS':
            layers_indexes = MUTABLE_LCHANGEPARAM_INDEXES #Only this indexes may change their parameters
            mutation_function = mutator_obj.mutate_layer_parameters
        elif mutation_type == 'L_CHANGE_TYPE':
            layers_indexes = MUTABLE_LCHANGETYPE_INDEXES #Only this indexes may change type
            mutation_function = mutator_obj.mutate_layer_type
        else:
            print(f'ERROR: {mutation_type} Mutation type not recognized')
            mutation_function = None
        for layer_idx in layers_indexes:
            if random.random() < self.MUT_PROB:
                mutated_layer, layer_type = mutation_function(arch_obj_ind.genotype, layer_idx)
                if mutation_type == 'L_CHANGE_TYPE':
                    #Change CONV for MAXPOOL, etc.
                    #layer_type is the original type. I have to change it.
                    old_layer_type = list(mutated_layer.keys())[0]
                    mutated_layer[layer_type] = mutated_layer.pop(old_layer_type)
                arch_obj_ind.genotype.gen_list[layer_idx] = mutated_layer
                arch_obj_ind.set_genotype(arch_obj_ind.genotype)
        
        return arch_obj_ind

    def mutate_children(self, mutation_type):
        if self.search_strategy == 'RANDOM':
            return
        
        for i in range(len(self.children)):
            name_before_mut = self.children[i].idx
            mutated_child = self.change_layer_parameters_or_type(self.children[i], mutation_type)
            mutated_child.idx = str(name_before_mut) + '[M]' #Add M to the index to indicate it was mutated      
            mutated_child = self.make_mutant(mutated_child, self.children[i])
            mutated_child = self.train_model(mutated_child)
            self.children[i] = copy.deepcopy(mutated_child)
    
    def crossover(self, arch_obj1, arch_obj2):
        crossover_obj = Crossover()
        #Check what kind of crossover is being used
        if crossover_type == 'SPC':
            point = random.choice(SPC_INDEXES)
            child1_arch, child2_arch = crossover_obj.single_point_crossover(arch_obj1, arch_obj2, point)
        elif crossover_type == 'TPC':
            [point1, point2] = random.sample(SPC_INDEXES, 2)
            child1_arch, child2_arch = crossover_obj.two_point_crossover(arch_obj1, arch_obj2, point1, point2)
        elif crossover_type == 'UC':
            child1_arch, child2_arch = crossover_obj.uniform_crossover(arch_obj1, arch_obj2)
        else:
            print('UNRECOGNIZED CROSSOVER METHOD')
            return None, None
        return child1_arch, child2_arch

    def random_parent_selection(self):
        [p1, p2] = random.sample(self.pop, 2)
        p1 = self.make_parent(p1)
        p2 = self.make_parent(p2)
        return (p1, p2)

    def generate_offspring(self):
        reporter = ReportENAS()
        #Use Crossover or Random search
        self.children = []
        if self.search_strategy == 'RANDOM':
            print(f'Generating and training random children')
            for i in range(self.NPOP):
                rand_arch = self.random_individual()
                rand_arch = self.train_model(rand_arch)
                self.children.append(rand_arch)
            print(f'\nDone\n')
                                
        elif self.search_strategy == 'GA':
            print(f'GA SEARCH STRATEGY')
            for _ in range(self.NPOP//2): #//2 because you add two children
                print('\nSelecting parents')
                parent1, parent2 = self.random_parent_selection()
                print('Done\n')
                print(f'Using crossover')
                child1, child2 = self.crossover(parent1, parent2)
                child1.genotype.gen_list = self.validate_architecture(child1.genotype.gen_list)
                child2.genotype.gen_list = self.validate_architecture(child2.genotype.gen_list)
                child1.set_genoStr()
                child2.set_genoStr()
                child1 = self.make_child(child1, parent1, parent2)
                child2 = self.make_child(child2, parent1, parent2)
                print('Done\n')
                
                print('Training parents and children')
                parent1 = self.train_model(parent1)
                parent2 = self.train_model(parent2)
                child1 = self.train_model(child1)
                child2 = self.train_model(child2)
                print('Parent and children training done')
                
                self.children.append(child1)
                self.children.append(child2)
                
    def create_model(self, arch_obj):
        layers_list = arch_obj.decode()
        model = models.Sequential(layers_list)
        return model

    def random_individual(self):
        # TODO: Add more representations and encodings.
        #Create the inner layers, then add the input size at 0 and the last layers at the end
        '''
        gen_list = [
                    {'INP':INPUT_SIZE},
                    create_conv_layer(),
                    create_conv_layer(),
                    create_pool_max_layer(),
                    create_conv_layer(),
                    create_conv_layer(),
                    create_pool_avg_layer(),
                    {'FLATTEN':None},
                    create_dense_layer(),
                    create_dense_layer(10, 'softmax')
                ]
        '''
        gen_list = []
        for _ in range(SIZE_GENLIST-NUM_FIXED_LAYERS):
            layer_type = random.choice(type_mutable_layers)
            layer_func = create_layers_functions_dict[layer_type]
            gen_list.append(layer_func())
        gen_list.insert(0, {'INP':INPUT_SIZE})
        gen_list.insert(1, create_conv_layer(32,3))
        gen_list.append(create_pool_max_layer())
        gen_list.append({'FLATTEN':None})
        gen_list.append(create_dense_layer())
        gen_list.append(create_dense_layer(256, 'relu'))
        gen_list.append(create_dense_layer(10, 'softmax'))
        gen_list = self.validate_architecture(gen_list)
        genotype = Genotype('L', 'IV', gen_list)
        idx = random.choice(ARCH_NAMES_LIST)
        arch_obj = LayerRepresentation('S', str(idx), genotype)
        
        return arch_obj

    def train_model(self, arch_obj, epochs = EPOCHS):
        print(Fore.GREEN + f'\n{ast} EXECUTION {e+1}/{EXECUTIONS} GENERATION {g+1}/{GENERATIONS} {crossover_type} {mutation_type} {self.arch_count}/{TOTAL_ARCH} architectures {ast}\n')
        print(Style.RESET_ALL)
        print(f'\nTraining {arch_obj.idx}...')
        isFather = False

        #Check if the architecture was already trained
        #Mutants become parents, but their information in the dictionary are as child. That's why I have to be sure they are parents in the new generation.
        if arch_obj.idx in list(self.trained_archs.keys()):
                print(Fore.YELLOW + f'\nArchitecture {arch_obj.idx} was already trained.')
                print(Style.RESET_ALL)
                self.arch_count += 1
                if arch_obj.isChild == False and arch_obj.isMutant == False: #IF is a parent
                    isFather = True
                arch = self.trained_archs[arch_obj.idx]
                if isFather == True:
                    arch = self.make_parent(arch)
                self.reporter.save_arch_info(arch, self.search_strategy, self.crossover_type, self.mutation_type, g, e, local_seed)
                return copy.deepcopy(arch)
               
        if TRAIN == False:
            print(f"Simulating trainining model {arch_obj.idx}")
            self.reporter.save_arch_info(arch_obj, self.search_strategy, crossover_type, mutation_type, g, e, local_seed)
            print(f'Simulating Training {arch_obj.idx} complete')
            self.trained_archs[arch_obj.idx] = arch_obj
            self.arch_count += 1
            return arch_obj

        # Normalize the images to [0, 1] range
        x_train, x_test = self.x_train / 255.0, self.x_test / 255.0
        y_train, y_test = self.y_train, self.y_test
        with tf.device('/gpu:0'):
            model = self.create_model(arch_obj)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            if epochs == 5:
                patience = 3
            else:
                patience = 5
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
            start_time = time.time()
            # Train the model
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))  # Adjust epochs as needed ==========================================
            end_time = time.time()

            # Evaluate the model on test data
            print('Evaluating model on test data...')
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print(f'\nTraining {arch_obj.idx} complete')

        # Calculate and print CPU-Hours
        training_time_seconds = end_time - start_time
        training_time_hours = training_time_seconds / 3600

        # Calculate the number of parameters
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        # Update arch_obj attributes

        arch_obj.acc_hist = history.history['val_accuracy']
        arch_obj.loss_hist = history.history['val_loss']
        arch_obj.acc = history.history['val_accuracy'][-1]  # Final validation accuracy
        arch_obj.loss = history.history['val_loss'][-1]     # Final validation loss
        arch_obj.cpu_hours = training_time_hours            # Training time in CPU-hours
        arch_obj.num_params = total_params                  # Total number of model parameters
        arch_obj.flops = calculate_model_flops(model)

        self.reporter.save_arch_info(arch_obj, self.search_strategy, self.crossover_type, self.mutation_type, g, e, local_seed)
        self.trained_archs[arch_obj.idx] = arch_obj
        self.arch_count += 1
        
        return arch_obj

    def ENAS(self):
        global SEED
        global local_seed
        global arch_count
        ast = 50*'+'
        global search_strategy
        global crossover_types_list, mutation_types_list
        for search_strategy in search_strategies_list:
            self.search_strategy = search_strategy
            if self.search_strategy == 'RANDOM':
                crossover_types_list = ['NONE']
                mutation_types_list = ['NONE']

            global crossover_type
            for crossover_type in crossover_types_list:
                self.crossover_type = crossover_type
                global mutation_type
                for mutation_type in mutation_types_list:
                    self.mutation_type = mutation_type
                    
                    global e
                    local_seed = SEED
                    #random.seed(local_seed)
                    for e in range(EXECUTIONS):
                        self.trained_archs = {}
                        random.seed(local_seed)
                        np.random.seed(local_seed)
                        #torch.manual_seed(local_seed)
                        print('\nInitializing population')
                        self.initialize_pop()
                        print('Done\n')
                        global g
                        for g in range(GENERATIONS):
                            if self.search_strategy == 'GA':
                                print(f'\nGenerating offspring {self.search_strategy}')
                                self.generate_offspring() #RANDOM OR CROSSOVER
                                print('\nMutating offspring')
                                self.mutate_children(mutation_type)
                                print('\nMutating children done')
                            else: #RANDOM SEARCH
                                self.children = []
                                for ind in self.pop:
                                    ind = self.train_model(ind)
                                    self.children.append(ind)

                            print('\nSelecting children as new population')
                            self.pop = self.children
                            print('\nDone...')
                            
                            #Sort all elements in the population to get the best accuracy
                            print('\nSorting elements to find best and median architecture')
                            sorted_pop = sorted(self.pop, key=lambda obj: obj.acc, reverse=True)
                            print('\nDone')
                            print('\nBest architecture is: ')
                            print(sorted_pop[0])
                        local_seed += 1
                        print('\nSaving histories')
                        self.accuracy_histories.append(sorted_pop[0].acc_hist)
                        self.loss_histories.append(sorted_pop[0].loss_hist)

                        self.best_acc_list.append(sorted_pop[0].acc)
                        self.loss_list.append(sorted_pop[0].loss)
                        self.cpu_hours_list.append(sorted_pop[0].cpu_hours)
                        self.num_params_list.append(sorted_pop[0].num_params)
                        self.flops_list.append(sorted_pop[0].flops)
                        self.best_archs.append(sorted_pop[0])
                        print('\nDone')
                    median_arch, idx = find_median(self.best_archs)
                    print('\nMedian architecture is: ')
                    print(median_arch)
                    #self.create_report(reporting_single_arch = True, single_arch = median_arch)  #CHANGE THIS ============================================
                    #self.create_report(reporting_single_arch = False)
                    if PLOT == True:
                        plotter = PlotterENAS(self)
                        plotter.plot_accuracy_loss_histories('L')
                        plotter.plot_accuracy_loss_histories('A')

    def initialize_pop(self):
        self.pop = []
        for i in range(self.NPOP):
            self.pop.append(self.random_individual())

    def __init__(self):
        self.NPOP = MAIN_NPOP
        self.MUT_PROB = MUT_PROB
        self.crossover_types_list = crossover_types_list
        self.crossover_type = None
        self.mutation_type = None
        self.search_strategy = None
        self.reporter = ReportENAS()
        self.arch_count = 0
        
        self.pop = []
        self.children = []
        self.trained_archs = {}
        self.accuracy_histories = [] #To plot all accuracies through epochs
        self.loss_histories = [] #To plot all accuracies through epochs
        self.get_normalize_dataset()

        self.report_columns = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs",
                   "Acc_mean", "Loss_mean", "CPU_hrs_mean", "Num_params_mean",
                   "FLOPs_mean", "Acc_std", "Loss_std", "CPU_hrs_std", "Num_params_std",
                   "FLOPs_std"]
        self.bar_columns = ['Acc_mean', "Loss_mean", "CPU_hrs_mean", "Num_params_mean", "FLOPs_mean"]
        self.box_columns = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs"]
        self.bar_median_columns = ['Acc_mean', "Loss_mean"]
        self.box_median_columns = ["Best_accuracy", "Loss"]
        #Lists for report
        self.best_accHist_list = []
        self.best_acc_list = []
        self.loss_list = []
        self.cpu_hours_list = []
        self.num_params_list = []
        self.flops_list = []

        self.best_archs = [] #For finding the median
        self.accuracy_median_histories = [] #To plot all accuracies through epochs, median arch
        self.loss_median_histories =[]

        self.general_report_filenames = [f'GA_L_CHANGE_TYPE', f'GA_L_MODIFY_PARAMS', f'RANDOM_NONE']
        self.medians_report_filenames = [f'GA_L_CHANGE_TYPE_MEDIAN', f'GA_L_MODIFY_PARAMS_MEDIAN', f'RANDOM_NONE_MEDIAN']
        self.filename = f'{self.search_strategy}_mutation_type' #For report and plot filenames


'''
gen_list = [{'INP': 32}, {'POOLMAX': [-1, 3]}, {'POOLAVG': [-1, 3]}, {'CONV': [128, 5]}, {'CONV': [64, 3]}, {'POOLMAX': [-1, 3]}, {'POOLMAX': [-1, 3]}, {'FLATTEN': None}, {'DENSE': [64, 'relu']}, {'DENSE': [10, 'softmax']}]
gen_obj = Genotype('L','I',gen_list)
arch = LayerRepresentation('S', genotype=gen_obj)
print()
print(arch.genotype.gen_list)
tecnas = TECNAS()
arch.genotype.gen_list = tecnas.validate_architecture(arch.genotype.gen_list)
print(arch.genotype.gen_list)
'''

'''
random.seed(1)
os.system("cls")

N = 2
tecnas = TECNAS()

archs = []
for _ in range(N):
    arch = tecnas.random_individual()
    archs.append(arch)
   
print()
for arch in archs:
    print(arch.genotype.gen_list)

cross = Crossover()
c1, c2 = cross.uniform_crossover(archs[0], archs[1])
print()
print(c1.genotype.gen_list)
print(c2.genotype.gen_list)
'''