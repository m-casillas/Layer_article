import psutil
import math
import tensorflow as tf
from colorama import Fore, Back, Style, init
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
#from tensorflow.python.profiler.model_analyzer import profile
#from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from globalsENAS import *
from Surrogate_ENAS import Surrogate_ENAS
from configENAS import *
from Genotype import *
from ReportENAS import *
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
    def simulate_training(self, arch_obj):
        # Simulate training and return the architecture with simulated accuracy and loss
        arch_obj.acc = random.randint(0, 100)
        arch_obj.loss = random.uniform(0, 1)
        arch_obj.flops = random.randint(0, 100)
        arch_obj.cpu_hours = random.uniform(0, 1)
        arch_obj.num_params = random.randint(0, 100)
        return arch_obj

    def GA_or_RANDOM(self):
        if self.search_strategy == 'GA':
            if self.crossover_type != 'NONE':
                print(f'\nGenerating offspring {self.search_strategy}')
                self.generate_offspring() #CROSSOVER
            else:
                #Copy the parents to the children in case of only mutation. (Mutation only works with children and they are generated in the crossover function)
                self.children = copy.deepcopy(self.pop)
                
            if self.mutation_type != 'NONE': 
                print('\nMutating offspring')
                self.mutate_children(self.mutation_type)
                print('\nMutating children done')
        else: #RANDOM SEARCH
            print(f'\nGenerating population {self.search_strategy}')
            self.children = []
            self.initialize_pop()
                        

    def child_better_parents(self, child):
        #Check if the child is better than both parents
        if child.acc > child.parent1.acc and child.acc > child.parent2.acc:
            return True
        else:
            return False

    def get_normalize_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        total_train_samples = x_train.shape[0]
        
        if DATASET_PART == 1:
            subset_x_train = x_train
            subset_y_train = y_train
        else:
            subset_x_train,_,subset_y_train,_ = train_test_split(x_train, y_train, train_size = DATASET_PART, stratify = y_train, random_state = 42)
        subset_y_train = subset_y_train.reshape(-1, 1)
        unique, counts = np.unique(subset_y_train, return_counts=True)
        print(dict(zip(unique, counts)))
        self.x_train = subset_x_train / 255.0
        self.x_test = x_test / 255.0
        self.y_train = subset_y_train.flatten()
        self.y_test = y_test.flatten()
        self.train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        self.train_datagen.fit(x_train)
        self.train_generator = self.train_datagen.flow(self.x_train, self.y_train, batch_size=BATCH_SIZE)

        self.validation_datagen = ImageDataGenerator()
        self.validation_generator = self.validation_datagen.flow(self.x_test, self.y_test, batch_size=BATCH_SIZE)


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

    def validate_architecture(self, gen_list, arch_idx = ''):
        #Determine if the architecture is valid before training. Check that there are no consecutive POOL layers
        #Returns a valid gen_list by changing the POOL to CONV
        print(f'Validating {arch_idx}:  {gen_list}')
        if len(gen_list) != SIZE_GENLIST:
            print(f'FATAL ERROR. ARCHITECTURE SIZE CHANGED: {len(gen_list)}, should be {SIZE_GENLIST}')
            print('Creating random architecture')
            return self.random_genlist()
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

        if mutation_type == 'MPARAMS':
            layers_indexes = MUTABLE_LCHANGEPARAM_INDEXES #Only this indexes may change their parameters
            mutation_function = mutator_obj.mutate_layer_parameters
        elif mutation_type == 'MTYPE':
            layers_indexes = MUTABLE_LCHANGETYPE_INDEXES #Only this indexes may change type
            mutation_function = mutator_obj.mutate_layer_type
        else:
            print(f'ERROR: {mutation_type} Mutation type not recognized')
            mutation_function = None
        for layer_idx in layers_indexes:
            if random.random() < self.MUT_PROB:
                mutated_layer, layer_type = mutation_function(arch_obj_ind.genotype, layer_idx)
                if mutation_type == 'MTYPE':
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
            self.total_mut += 1
            mutated_child.idx = str(name_before_mut) + '[M]' #Add M to the index to indicate it was mutated      
            mutated_child = self.make_mutant(mutated_child, self.children[i])
            mutated_child = self.train_model(mutated_child)
            #Assesing if the mutation was succesful
            self.succ_mut += 1 if mutated_child.acc > self.children[i].acc else 0
            self.children[i] = copy.deepcopy(mutated_child)
    
    def crossover(self, arch_obj1, arch_obj2):
        crossover_obj = Crossover()
        #Check what kind of crossover is being used
        if self.crossover_type == 'SPC':
            point = random.choice(SPC_INDEXES)
            child1_arch, child2_arch = crossover_obj.single_point_crossover(arch_obj1, arch_obj2, point)
        elif self.crossover_type == 'TPC':
            [point1, point2] = random.sample(SPC_INDEXES, 2)
            child1_arch, child2_arch = crossover_obj.two_point_crossover(arch_obj1, arch_obj2, point1, point2)
        elif self.crossover_type == 'UC':
            child1_arch, child2_arch = crossover_obj.uniform_crossover(arch_obj1, arch_obj2)
        else:
            print(f'UNRECOGNIZED CROSSOVER METHOD {self.crossover_type}')
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
            self.total_cross += 2
            print('Done\n')
            
            print('Training children')
            child1 = self.train_model(child1)
            child2 = self.train_model(child2)
            print('Children training done')

            print('\nAssessing if crossover was succesful')
            self.succ_cross += 1 if self.child_better_parents(child1) else 0
            self.succ_cross += 1 if self.child_better_parents(child2) else 0
            print('Done\n')
            
            self.children.append(child1)
            self.children.append(child2)
            
                
    def create_model(self, arch_obj):
        layers_list = arch_obj.decode()
        model = models.Sequential(layers_list)
        #model.summary()
        return model
    
    def random_genlist(self):
        pool_count = 0
        max_pool_count = 3
        gen_list = []
        for _ in range(SIZE_GENLIST-NUM_FIXED_LAYERS):
            layer_type = random.choice(type_mutable_layers)
            if layer_type == 'POOLMAX' or layer_type == 'POOLAVG':
                pool_count += 1
                if pool_count > max_pool_count:
                    layer_type = 'CONV'

            layer_func = create_layers_functions_dict[layer_type]
            gen_list.append(layer_func())
        gen_list.insert(0, {'INP':INPUT_SIZE})
        gen_list.insert(1, create_conv_layer())
        gen_list.append({'FLATTEN':None})
        gen_list.append(create_dense_layer())
        gen_list.append(create_dense_layer(10, 'softmax'))
        gen_list = self.validate_architecture(gen_list)
        return gen_list

    def random_individual(self):
        # TODO: Add more representations and encodings.
        #Create the inner layers, then add the input size at 0 and the last layers at the end
        
        gen_list = []
        gen_list = self.random_genlist()
        genotype = Genotype('L', 'IV', gen_list)
        idx = random.choice(ARCH_NAMES_LIST)
        arch_obj = LayerRepresentation('S', str(idx), genotype)
        
        return arch_obj

    def train_model(self, arch_obj, epochs = EPOCHS):
        ast = self.ast
        print(Fore.BLUE + f'\n{ast} EXECUTION {self.exec}/{EXECUTIONS} GENERATION {self.generation}/{GENERATIONS} {self.crossover_type} {self.mutation_type} {self.arch_count+1}/{TOTAL_ARCH} architectures {ast}\n' + Style.RESET_ALL)
        self.arch_count += 1
        memory = psutil.virtual_memory()
        # Print the percentage of RAM used
        print(Fore.RED + f"RAM used: {memory.percent}%")
        print(Style.RESET_ALL)
        print(f'\nTraining {arch_obj.idx}...')
        print(arch_obj.genotype.gen_list)
        isFather = False

        #Check if the architecture was already trained
        #Mutants become parents, but their information in the dictionary are as child. That's why I have to be sure they are parents in the new generation.
        if arch_obj.idx in list(self.trained_archs.keys()):
                print(Fore.YELLOW + f'\nArchitecture {arch_obj.idx} was already trained.')
                print(Style.RESET_ALL)
                
                if arch_obj.isChild == False and arch_obj.isMutant == False: #IF is a parent
                    isFather = True
                arch = self.trained_archs[arch_obj.idx]
                if isFather == True:
                    arch = self.make_parent(arch)
                self.reporter.save_arch_info(self, arch)
                return copy.deepcopy(arch)
        
        if SIMULATE == True:
            print(f"Simulating trainining model {arch_obj.idx}")
            print(arch_obj.genotype.gen_list)
            arch_obj = self.simulate_training(arch_obj)
            model = self.create_model(arch_obj)
            #arch_obj.num_params = calculate_model_params(model)
            #arch_obj.flops = calculate_model_flops(model)
            self.reporter.save_arch_info(self, arch_obj)
            print(f'Simulating Training {arch_obj.idx} complete. Accuracy {arch_obj.acc}\n')
            self.trained_archs[arch_obj.idx] = arch_obj
            return arch_obj
        
        if SURROGATE == True:
            print(f"Predicting model {arch_obj.idx}")
            print(arch_obj.genotype.gen_list)
            self.surrogate.load_arch(arch_obj)
            arch_obj.acc = self.surrogate.predict_arch(arch_obj, self.regressor_type)
            model = self.create_model(arch_obj)
            arch_obj.num_params = calculate_model_params(model) # Total number of model parameters
            arch_obj.flops = calculate_model_flops(model)
            arch_obj.acc_hist = []
            arch_obj.loss_hist = []
            arch_obj.loss = -99999     # Final validation loss
            arch_obj.cpu_hours = -99999            # Training time in CPU-hours
            arch_obj.num_params = calculate_model_params(model) # Total number of model parameters
            arch_obj.flops = calculate_model_flops(model)
            arch_obj.trained_epochs = -99999

            self.reporter.save_arch_info(self, arch_obj)
            self.trained_archs[arch_obj.idx] = arch_obj
            return arch_obj


        if TRAIN == True:
            with tf.device('/gpu:0'):
                while True:
                    try:
                        tf.keras.backend.clear_session()
                        model = self.create_model(arch_obj)
                        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        patience = 3 if epochs == 5 else 5
                        early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
                        #steps_per_epoch = math.ceil(len(self.x_train) / BATCH_SIZE)
                        start_time = time.time()
                        #history = model.fit(self.train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=self.validation_generator,  callbacks=[early_stopping] )
                        history = model.fit(self.train_generator, epochs=epochs, validation_data=self.validation_generator)
                        end_time = time.time()
                        # Evaluate the model on test data
                        print(f'Evaluating {arch_obj.idx} on test data...')
                        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
                        print(f'\nTraining {arch_obj.idx} complete')
                        break  # training successful, exit the loop
                    except Exception as e:
                        print(Fore.RED + f"Training failed for architecture {arch_obj.idx}: {e}")
                        print(Style.RESET_ALL)
                        print("Generating a new random architecture...")
                        arch_obj.genotype.gen_list = self.random_genlist()
                        arch_obj.wasInvalid = True
            # Calculate and print CPU-Hours
            training_time_seconds = end_time - start_time
            training_time_hours = training_time_seconds / 3600
            arch_obj.acc_hist = history.history['val_accuracy']
            arch_obj.loss_hist = history.history['val_loss']
            arch_obj.acc = history.history['val_accuracy'][-1]  # Final validation accuracy
            arch_obj.loss = history.history['val_loss'][-1]     # Final validation loss
            arch_obj.cpu_hours = training_time_hours            # Training time in CPU-hours
            arch_obj.num_params = calculate_model_params(model) # Total number of model parameters
            arch_obj.flops = calculate_model_flops(model)
            arch_obj.trained_epochs = len(history.history['loss'])

            self.reporter.save_arch_info(self, arch_obj)
            self.trained_archs[arch_obj.idx] = arch_obj
            return arch_obj

    def select_elitism(self):
        if self.search_strategy == 'RANDOM':
            #self.pop = self.children
            return
        
        self.pool = self.pop + self.children
        # Sort the population by accuracy
        sorted_pool = sorted(self.pool, key=lambda obj: obj.acc, reverse=True)
        # Select the top N individuals to keep in the next generation
        self.pop = sorted_pool[:self.NPOP]
               
    def ENAS(self):
        global crossover_types_list
        global mutation_types_list
        for search_strategy in search_strategies_list:
            self.search_strategy = search_strategy
            if self.search_strategy == 'RANDOM':
                crossover_types_list = ['NONE']
                mutation_types_list = ['NONE']
            #global crossover_type
            for crossover_type in crossover_types_list:
                self.crossover_type = crossover_type
                #global mutation_type
                for mutation_type in mutation_types_list:
                    self.mutation_type = mutation_type

                    if crossover_type == 'NONE' and mutation_type == 'NONE':
                        self.search_strategy = 'RANDOM'
                        self.crossover_type = 'NONE'
                        self.mutation_type = 'NONE'

                    if RANDOMIZE_SEED == True:
                        self.local_seed = random.randint(7,23357) #SEED
                    else:
                        self.local_seed = SEED
                    random.seed(self.local_seed)
                    np.random.seed(self.local_seed)
                    tf.random.set_seed(self.local_seed)

                    for e in range(1,EXECUTIONS+1):
                        self.exec = e
                        #self.trained_archs = {}
                        if self.search_strategy == 'GA':
                            print('\nInitializing population')
                            self.generation = 1
                            self.initialize_pop()
                            print('Initialization Done\n')
                        self.succ_cross = 0
                        self.succ_mut = 0
                        self.total_cross = 0
                        self.total_mut = 0
                        for g in range(1,GENERATIONS+1):
                            self.pool = []
                            self.generation = g
                            self.GA_or_RANDOM()
                            print('\nSelecting best parents and children for the new population')
                            self.select_elitism()
                            print('Done...')
                            print('\nSorting elements to find best and median architecture')
                            sorted_pop = sorted(self.pop, key=lambda obj: obj.acc, reverse=True)
                            print('Done Selecting')
                            print('\nBest architecture is: ')
                            print(sorted_pop[0].idx, sorted_pop[0].acc)
                            sorted_pop[0].bestGen = True
                            reporter = ReportENAS()
                            #Save the best architecture in the report
                            reporter.save_arch_info(self, sorted_pop[0], isBest = True)
                            sorted_pop[0].bestGen = False
                            
                        self.local_seed += 1
                        print('\nSaving histories')
                        self.accuracy_histories.append(sorted_pop[0].acc_hist)
                        self.loss_histories.append(sorted_pop[0].loss_hist)

                        self.best_acc_list.append(sorted_pop[0].acc)
                        self.loss_list.append(sorted_pop[0].loss)
                        self.cpu_hours_list.append(sorted_pop[0].cpu_hours)
                        self.num_params_list.append(sorted_pop[0].num_params)
                        self.flops_list.append(sorted_pop[0].flops)
                        self.best_archs.append(sorted_pop[0])
                        print('Done========================')
                    #median_arch, idx = find_median(self.best_archs)
                    #print('\nMedian architecture is: ')
                    #print(median_arch)
                    #self.create_report(reporting_single_arch = True, single_arch = median_arch)  #CHANGE THIS ============================================
                    #self.create_report(reporting_single_arch = False)
                    
    def initialize_pop(self):
        self.pop = []
        self.pool = []
        print('Generating and training random architectures')
        for i in range(self.NPOP):
            random_arch = self.random_individual()
            random_arch = self.train_model(random_arch)
            self.pop.append(random_arch)
        print('Done')

    def __init__(self, regressor_type = -1):
        self.ast = 50*'+'
        self.NPOP = MAIN_NPOP
        self.MUT_PROB = MUT_PROB
        self.crossover_types_list = crossover_types_list
        self.crossover_type = None
        self.mutation_type = None
        self.search_strategy = None
        self.reporter = ReportENAS()
        self.generation = 0
        self.exect = 0
        self.arch_count = 0
        self.succ_mut = 0
        self.succ_cross = 0
        self.total_mut = 0
        self.total_cross = 0
        self.pop = []
        self.children = []
        self.pool = [] #Parents + children
        self.trained_parents = []
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

        self.regressor_type = regressor_type
        self.regressor_folder = os.path.join(path, 'results', 'surrogates')
        if self.regressor_type >= 0:
            self.surrogate = Surrogate_ENAS(self.regressor_type, self.regressor_folder)

