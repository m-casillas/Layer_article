import tensorflow as tf
from colorama import Fore, Back, Style, init
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input
#from globalsENAS import *
from configENAS import *
from Genotype import *
from ReportENAS import *
from PlotterENAS import *
from LayerRepresentation import *
from Crossover import *
from Mutator import *
from TECNAS import *
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

class Trainer:
    def create_model(self, arch_obj):
            layers_list = arch_obj.decode()
            model = models.Sequential(layers_list)
            return model

    def validate_architecture(self, gen_list):
        #Determine if the architecture is valid before training. Check that there are no consecutive POOL layers
        #Returns a valid gen_list by changing the POOL to CONV
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
     
    def train(self, arch_obj, epochs = EPOCHS):
            # Normalize the images to [0, 1] range
            x_train, x_test = self.x_train / 255.0, self.x_test / 255.0
            y_train, y_test = self.y_train, self.y_test
            print(f'\nValidating {arch_obj.idx}')
            print(f'{arch_obj.genotype.gen_list}')
            arch_obj.genotype.gen_list = self.validate_architecture(arch_obj.genotype.gen_list)
            arch_obj.set_genoStr()
            
            if TRAIN == False:
                print(f"Simulating trainining model {arch_obj.idx}")
                print(f'Simulating Training {arch_obj.idx} complete')
                self.reporter = ReportENAS()
                self.reporter.save_arch_info(arch_obj, '', '', 0, 0, 0, epochs)
                return arch_obj
            with tf.device('/gpu:0'):
                model = self.create_model(arch_obj)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                if epochs == 5:
                     patience = 3
                else:
                     patience = 5
                early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
                print(f"Trainining model {arch_obj.idx}")
                start_time = time.time()
                # Train the model
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=[early_stopping])  # Adjust epochs as needed ==========================================
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
            self.reporter = ReportENAS()
            self.reporter.save_arch_info(arch_obj, '', '', 0, 0, 0, epochs)
            return arch_obj
    
    def __init__(self):
         #Load a portion of the dataset, given by DATASET_PART
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()
        total_train_samples = self.x_train.shape[0]
        if DATASET_PART == 1:
            # Use the whole dataset
            middle_start = 0
            middle_end = total_train_samples
        else:
            # Use a subset of the dataset
            middle_start = total_train_samples // DATASET_PART
            middle_end = 3 * total_train_samples // DATASET_PART


N = 30 #Number of architectures
EPOCHS_LIST = [5, 10, 20, 100]
random.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.system("cls")
tecnas = TECNAS()
trainer = Trainer()

archs = []
for _ in range(N):
    arch = tecnas.random_individual()
    archs.append(arch)

for arch in archs:
     print(arch.genotype.gen_list)

for epochs in EPOCHS_LIST:
    for i,arch in enumerate(archs):   
        print(f'\n@@@@@@@@@@@@@@@@@@@\nEpochs: {epochs} Arch {i+1}/{N}\n@@@@@@@@@@@@@@@@@@@\n')
        trainer.train(arch, epochs = epochs)
