import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.layers import Input
from globalsENAS import *
from utilitiesENAS import *
from configENAS import *
from Genotype import *
from LayerRepresentation import *
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""# TECNAS Classs"""

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#genotype = Genotype('L', 'IV', gen_list)
#arch = Architecture('X', 0, 0, 0, 0, 1, genotype)

#TECNAS is a class that performs evoltuionary neural architecture search, using genetic operators
class TECNAS:
    #check_pool_pool determines if there are two consecutive MAXPOOL layers, and change the second one for a CONV. WORKING ON THIS ==================================
    def check_pool_pool(self, current_layer_index):
        genotype = self.pop[0].genotype.gen_list  # Assuming we're modifying the first architecture in the population
        if current_layer_index < len(genotype) - 1:  # Ensure there's a next layer
            current_layer = genotype[current_layer_index]
            next_layer = genotype[current_layer_index + 1]
            if 'POOLMAX' in next_layer:  # Check if the next layer is 'POOLMAX'
                new_conv_layer = create_conv_layer()  # Create a CONV layer
                genotype[current_layer_index + 1] = new_conv_layer  # Replace the next layer
                print(f"Replaced layer {current_layer_index + 1} (POOLMAX) with {new_conv_layer}.")

    def create_report(self, reporting_single_arch = False, single_arch = None):
        execList = list(range(1,EXECUTIONS+1))
        epochsList = list(range(1, EPOCHS+1))

        if reporting_single_arch == True: #Print the median arch information
            empty_list = ['']*(len(epochsList)-1)
            data = pd.DataFrame({   f"Epochs":epochsList,  f"Best_accuracy": single_arch.acc_hist,
                                    f"Loss": single_arch.loss_hist,  f"Acc_mean": [np.mean(single_arch.acc_hist)]+empty_list,
                                    f"Loss_mean": [np.mean(single_arch.loss_hist)]+empty_list
                                })
            path_report = os.path.join(path_results, f'{self.filename}_MEDIAN.csv')
        else:
            empty_list = ['']*(len(execList)-1)
            data = pd.DataFrame({   f"Execution":execList,  f"Best_accuracy": self.best_acc_list,
                                    f"Loss": self.loss_list,  f'CPU_hrs':self.cpu_hours_list,
                                    f'Num_params':self.num_params_list, f'FLOPs':self.flops_list,
                                    f"Acc_mean": [np.mean(self.best_acc_list)]+empty_list, f"Loss_mean": [np.mean(self.loss_list)]+empty_list,
                                    f'CPU_hrs_mean':[np.mean(self.cpu_hours_list)]+empty_list, f'Num_params_mean':[np.mean(self.num_params_list)]+empty_list,
                                    f'FLOPs_mean':[np.mean(self.flops_list)]+empty_list, f"Acc_std": [np.std(self.best_acc_list)]+empty_list,
                                    f"Loss_std": [np.std(self.loss_list)]+empty_list, f'CPU_hrs_std':[np.std(self.cpu_hours_list)]+empty_list,
                                    f'Num_params_std':[np.std(self.num_params_list)]+empty_list, f'FLOPs_std':[np.std(self.flops_list)]+empty_list
                                })

            path_report = os.path.join(path_results, f'{self.filename}.csv')
        print(f'Report saved in {path_report}')
        data.to_csv(path_report, index=False)

    def change_layer_parameters_or_type(self, arch_obj_ind, mutation_type):
        #gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
        #Mutate each layer with probability MUT_PROB.
        #Save it in the genotype and update it.
        #Return the mutated architecture
        mutator_obj = Mutator()

        if mutation_type == 'L_MODIFY_PARAMS':
            layers_indexes = [1,2,3,4,6] #Only this indexes may change their parameters
            mutation_function = mutator_obj.mutate_layer_parameters
        elif mutation_type == 'L_CHANGE_TYPE':
            layers_indexes = [1,2,3,4] #Only this indexes may change type
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
        for i in range(len(self.children)):
            mutated_child = self.change_layer_parameters_or_type(self.children[i], mutation_type)
            self.children[i] = mutated_child

    def crossover(self, arch_obj1, arch_obj2):
        crossover_obj = Crossover()
        child1_layers_list, child2_layers_list = crossover_obj.single_point_crossover(arch_obj1, arch_obj2)
        return child1_layers_list, child2_layers_list

    def random_parent_selection(self):
        return random.sample(self.pop, 2)

    def generate_offspring(self):
        #Use Crossover or Random search
        self.children = []
        if self.search_strategy == 'RANDOM':
            print(f'RANDOM SEARCH STRATEGY')
            for _ in range(self.NPOP):
                self.children.append(self.random_individual())
        elif self.search_strategy == 'GA':
            print(f'GA SEARCH STRATEGY')
            for _ in range(self.NPOP//2): #//2 because you add two children
                parent1, parent2 = self.random_parent_selection()
                child1, child2 = self.crossover(parent1, parent2)
                self.children.append(child1)
                self.children.append(child2)

    def create_model(self, arch_obj):
        layers_list = arch_obj.decode()
        model = models.Sequential(layers_list)
        return model

    def random_individual(self):
        # TODO: Add more representations and encodings.
        gen_list = [{'INP':32},
                    {'CONV':[np.random.randint(NUM_FILTERS[0],NUM_FILTERS[1]+1), np.random.randint(CONV_KERN[0],CONV_KERN[1]+1)]},
                    create_pool_layer(),
                    {'CONV':[np.random.randint(NUM_FILTERS[0],NUM_FILTERS[1]+1), np.random.randint(CONV_KERN[0],CONV_KERN[1]+1)]},
                    {'POOLMAX':np.random.randint(POOL_KERN[0],POOL_KERN[1]+1)}, # CHECK THIS
                    {'FLATTEN':None},
                    {'DENSE':[np.random.randint(DENSE_NEURONS[0],DENSE_NEURONS[1]+1),'relu']},
                    {'DENSE':[10,'softmax']}]

        genotype = Genotype('L', 'IV', gen_list)
        idx = random.randint(0,1000)
        arch_obj = LayerRepresentation('S', str(idx), genotype)
        return arch_obj

    def train_model(self, arch_obj):
        # Normalize the images to [0, 1] range
        x_train, x_test = self.x_train / 255.0, self.x_test / 255.0
        y_train, y_test = self.y_train, self.y_test
        # Expand the dimensions to include the channel (since MNIST is grayscale, we have a single channel)
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        success = False
        max_attempts = 5  # Maximum number of retries
        attempts = 0

        while not success and attempts < max_attempts:
            try:
                # Attempt to create and train the model
                print(f"Attempt {attempts + 1} to train the model...")
                model = self.create_model(arch_obj)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                start_time = time.time()
                # Train the model
                print('Training model...')
                history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=256, validation_data=(x_test, y_test))  # Adjust epochs as needed ==========================================
                end_time = time.time()

                # Evaluate the model on test data
                print('Evaluating model on test data...')
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

                # If we reach this point, training was successful
                success = True
            except ValueError as ve:
                print(f"Model creation or training failed due to invalid architecture: {ve} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(arch_obj)
                self.change_layer_parameters_or_type(arch_obj, mutation_type)

            except Exception as e:
                print(f"An unexpected error occurred: {e} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(arch_obj)
                self.change_layer_parameters_or_type(arch_obj, mutation_type)

            finally:
                attempts += 1

        if not success:
            print("New architecture generated.")
            arch_obj = self.random_individual()  # Generate a new architecture for unexpected errors
            model = self.create_model(arch_obj)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            start_time = time.time()
            # Train the model
            print('Training model...')
            history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=256, validation_data=(x_test, y_test))  # Adjust epochs as needed ==========================================
            end_time = time.time()

            # Evaluate the model on test data
            print('Evaluating model on test data...')
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

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

    def ENAS(self):
        ast = 50*'+'
        for e in range(EXECUTIONS):
            self.initialize_pop()
            for g in range(GENERATIONS):
                #print(f'{ast} EXECUTION {e+1}/{EXECUTIONS} GENERATION {g+1}/{GENERATIONS} {ast}\n')
                print('Initializing population')
                print(f'Generating offspring {self.search_strategy}')
                self.generate_offspring() #RANDOM OR CROSSOVER
                #['L_MODIFY_PARAMS', 'L_CHANGE_TYPE']
                if self.search_strategy == 'GA':
                    print('Mutating offspring')
                    self.mutate_children(mutation_type)
                self.pop = self.children

                for i,arch in enumerate(self.pop):
                    print(f'\n{ast} {mutation_type} EXECUTION {e+1}/{EXECUTIONS} GENERATION {g+1}/{GENERATIONS} Training Architecture {i+1}/{self.NPOP} {ast}')
                    if TRAIN == True:
                        self.train_model(arch)
                #Sort all elements in the population to get the best accuracy
                sorted_pop = sorted(self.pop, key=lambda obj: obj.acc, reverse=True)
                print('Best architecture is: ')
                print(sorted_pop[0])
            self.accuracy_histories.append(sorted_pop[0].acc_hist)
            self.loss_histories.append(sorted_pop[0].loss_hist)

            self.best_acc_list.append(sorted_pop[0].acc)
            self.loss_list.append(sorted_pop[0].loss)
            self.cpu_hours_list.append(sorted_pop[0].cpu_hours)
            self.num_params_list.append(sorted_pop[0].num_params)
            self.flops_list.append(sorted_pop[0].flops)
            self.best_archs.append(sorted_pop[0])
        median_arch, idx = find_median(self.best_archs)
        print('\nMedian architecture is: ')
        print(median_arch)
        self.create_report(reporting_single_arch = True, single_arch = median_arch)
        self.create_report(reporting_single_arch = False)
        self.plot_accuracy_loss_histories('L')
        self.plot_accuracy_loss_histories('A')

    def initialize_pop(self):
        self.pop = []
        for i in range(self.NPOP):
            self.pop.append(self.random_individual())

    def __init__(self):
        self.search_strategy = search_strategy
        self.NPOP = MAIN_NPOP
        self.MUT_PROB = MUT_PROB
        
        self.pop = []
        self.children = []
        self.accuracy_histories = [] #To plot all accuracies through epochs
        self.loss_histories = [] #To plot all accuracies through epochs

        #Load a portion of the dataset, given by DATASET_PART
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()
        total_train_samples = self.x_train.shape[0]
        middle_start = total_train_samples // DATASET_PART
        middle_end = 3 * total_train_samples // DATASET_PART
        self.x_train = self.x_train[middle_start:middle_end]
        self.y_train = self.y_train[middle_start:middle_end]

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
        self.filename = f'{search_strategy}_{mutation_type}' #For report and plot filenames