from globalsENAS import *
from Architecture import *
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input

"""#Layer Representation Class"""

#Architecture is a deep neural network consisting in the input, conv layer, pooling layer, conv layer, pooling layer, fully connected layer
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class LayerRepresentation(Architecture):
    def genList_to_integer_vector(self):
        #Represents the gen list (list of dictionaries) as an integer vector. The input and the last Dense layer are not included.
        integer_vector = []

        #ARCH TO INT FUNCTION. ALSO NEED TO CREATE INT TO ARCH FUNCTION
        for dictio in self.genotype.gen_list:
            for layer_type in dictio.keys():
                layer_type_int = get_key_from_value(LAYERS_TYPES, layer_type)
                if layer_type in ['INP', 'FLATTEN'] :
                    continue
                elif layer_type == 'CONV':
                    integer_vector.append(layer_type_int)
                    nf = get_key_from_value(NUM_FILTERS, dictio[layer_type][0])
                    integer_vector.append(nf)
                    ks = get_key_from_value(CONV_KERNELS, dictio[layer_type][1])
                    integer_vector.append(ks)
                elif layer_type == 'POOLMAX':
                    integer_vector.append(layer_type_int)
                    nf = -1 #This must be -1
                    integer_vector.append(nf)
                    ks = get_key_from_value(POOL_KERNELS, dictio[layer_type][1])
                    integer_vector.append(ks)
                elif layer_type == 'POOLAVG':
                    integer_vector.append(layer_type_int)
                    nf = -1 #This must be -1
                    integer_vector.append(nf)
                    ks = get_key_from_value(POOL_KERNELS, dictio[layer_type][1])
                    integer_vector.append(ks)
                elif layer_type == 'DENSE':
                    nn = get_key_from_value(DENSE_NEURONS, dictio[layer_type][0])
                    if nn == 0: #It's the last DENSE layer, don't include it.
                        #print('genList_to_integer_vector: Last Dense layer, not included.')
                        continue
                    integer_vector.append(layer_type_int)
                    integer_vector.append(nn)
                    act = get_key_from_value(ACTIVATION_FUNCTIONS, dictio[layer_type][1])
                    integer_vector.append(act)
                else:
                    print('genList_to_integer_vector: Unidentified layer.')
                    integer_vector.append(-9)
        return integer_vector
    
    def integer_vector_to_genList(self, integer_vector):
        #Represents the integer vector as a gen list (list of dictionaries). The input and the last Dense layer are not included.
        gen_list = [{'INP':INPUT_SIZE}]
        for i in range(0, len(integer_vector), 3):
            layer_type = LAYERS_TYPES[integer_vector[i]]
            if layer_type == 'INP':
                continue
            elif layer_type == 'CONV':
                nf = list(NUM_FILTERS.values())[integer_vector[i+1]]
                ks = list(CONV_KERNELS.values())[integer_vector[i+2]]
                gen_list.append({'CONV':[nf, ks]})
            elif layer_type == 'POOLMAX':
                nf = -1
                ks = list(POOL_KERNELS.values())[integer_vector[i+2]]
                gen_list.append({'POOLMAX':[nf, ks]})
            elif layer_type == 'POOLAVG':
                nf = -1
                ks = list(POOL_KERNELS.values())[integer_vector[i+2]]
                gen_list.append({'POOLAVG':[nf, ks]})
            elif layer_type == 'FLATTEN':
                continue
            elif layer_type == 'DENSE':
                nn = list(DENSE_NEURONS.values())[integer_vector[i+1]]
                act = list(ACTIVATION_FUNCTIONS.values())[integer_vector[i+2]]
                gen_list.append({'DENSE':[nn, act]})
        #I want to append FLATTEN:None before the two last Dense layers
        gen_list.insert(-2, {'FLATTEN':None})
        return gen_list

    #Creates the architecture using the genotype information
    def decode_original(self):
        if self.genotype.rep_type == 'L':
            #This list stores each layer to create the Sequential model
            sequential_model = []
            for layer in self.genotype.gen_list:
                layer_type = list(layer.keys())[0] #Get the key of the dictionary "the layer type"
                if layer_type == 'INP':
                    input_size = layer[layer_type]
                    sequential_model.append(Input(shape=(input_size, input_size, 3)))
                elif layer_type == 'CONV':
                    num_filters = layer[layer_type][0]
                    kernel_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same'))
                elif layer_type == 'POOLMAX':
                    pool_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.MaxPooling2D(pool_size))
                elif layer_type == 'POOLAVG':
                    pool_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.AveragePooling2D(pool_size))
                elif layer_type == 'FLATTEN':
                    sequential_model.append(layers.Flatten())
                elif layer_type == 'DENSE':
                    num_neurons = layer[layer_type][0]
                    activation = layer[layer_type][1]
                    sequential_model.append(layers.Dense(num_neurons, activation=activation))
                else:
                    print(f'ERROR: {layer_type} Layer type not recognized')
                    sequential_model = None
        return sequential_model
    
    def decode(self):
        if self.genotype.rep_type == 'L':
        # This list stores each layer to create the Sequential model
            sequential_model = []
            for layer in self.genotype.gen_list:
                layer_type = list(layer.keys())[0]  # Get the key of the dictionary "the layer type"
                if layer_type == 'INP':
                    input_size = layer[layer_type]
                    sequential_model.append(Input(shape=(input_size, input_size, 3)))
                elif layer_type == 'CONV':
                    num_filters = layer[layer_type][0]
                    kernel_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.Conv2D(num_filters, kernel_size, padding='same'))
                    sequential_model.append(layers.BatchNormalization())
                    sequential_model.append(layers.Activation('relu'))
                    
                elif layer_type == 'POOLMAX':
                    pool_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.MaxPooling2D(pool_size, padding = 'same'))
                    sequential_model.append(layers.Dropout(0.3))
                elif layer_type == 'POOLAVG':
                    pool_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.AveragePooling2D(pool_size, padding = 'same'))
                    sequential_model.append(layers.Dropout(0.3))
                elif layer_type == 'FLATTEN':
                    sequential_model.append(layers.Flatten())
                elif layer_type == 'DENSE':
                    num_neurons = layer[layer_type][0]
                    activation = layer[layer_type][1]

                    if activation == 'softmax':
                        sequential_model.append(layers.Dense(num_neurons, activation=activation))
                   
                    else:
                        sequential_model.append(layers.Dense(num_neurons))
                        sequential_model.append(layers.BatchNormalization())
                        sequential_model.append(layers.Activation(activation))
                        sequential_model.append(layers.Dropout(0.5))
                    
                else:
                    print(f'ERROR: {layer_type} Layer type not recognized')
                    sequential_model = None
        return sequential_model

    def __init__(self, arch_type = 'S', idx = 9999, genotypeObj = None):
        super().__init__(arch_type, idx, genotypeObj)
        if is_None_or_empty(genotypeObj):
            self.integer_encoding = []
        else:
            self.integer_encoding = self.genList_to_integer_vector()
