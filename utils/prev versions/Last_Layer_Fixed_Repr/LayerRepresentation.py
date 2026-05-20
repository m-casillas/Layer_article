from globalsENAS import *
from Architecture import *

"""#Layer Representation Class"""

#Architecture is a deep neural network consisting in the input, conv layer, pooling layer, conv layer, pooling layer, fully connected layer
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class LayerRepresentation(Architecture):

    def integer_encoding_to_binary_encoding(self, maxInt):
        self.binary_encoding = []
        for layer_int in self.integer_encoding:
            layer_bin = int_to_gray(layer_int, maxInt)
            self.binary_encoding.append(layer_bin)

    def binary_encoding_to_integer_encoding(self):
        self.integer_encoding = []
        for layer_bin in self.binary_encoding:
            layer_int = gray_to_int(layer_bin)
            self.integer_encoding.append(layer_int)

    def genList_to_integer_vector(self):
        #Represents the gen list (list of dictionaries) as an integer vector. The input and the last Dense layer are not included.
        integer_vector = []
        for layer in self.genotype.gen_list:
            layer_type = get_key_from_dict(layer) #Get the key of the dictionary "the layer type"
            if layer_type in ['INP', 'FLATTEN', 'GLOBAL_AVG', 'LAST_DENSE']:
                continue
            else:
                layer_int = Globals.all_layers.index(layer)
                integer_vector.append(layer_int)
                
        return integer_vector
    
    def integer_vector_to_genList(self, integer_vector):
        #Represents the integer vector as a gen list (list of dictionaries). The input and the last Dense layer are not included.
        gen_list = [{'INP':Globals.INPUT_SIZE}]
        for layer_int in integer_vector:
            layer = Globals.all_layers[layer_int]
            gen_list.append(layer)

        #I want to append FLATTEN:None before the DENSE layer. LAST_DENSE hasn't been added yet.
        gen_list.insert(-1, {'FLATTEN':None})
        gen_list.append({'LAST_DENSE':[Globals.NUM_CLASSES,'softmax']})
        return gen_list

    
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
                    #sequential_model.append(layers.Conv2D(num_filters, kernel_size, padding='same'))
                    sequential_model.append(layers.Conv2D(num_filters, kernel_size = kernel_size, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4)))
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
                elif layer_type == 'GLOBAL_AVG':
                    sequential_model.append(layers.GlobalAveragePooling2D())
                    sequential_model.append(layers.Dropout(0.3))

                elif layer_type == 'FLATTEN':
                    sequential_model.append(layers.Flatten())
                elif layer_type == 'DENSE':
                    num_neurons = layer[layer_type][0]
                    activation = layer[layer_type][1]
                    sequential_model.append(layers.Dense(num_neurons))
                    sequential_model.append(layers.BatchNormalization())
                    sequential_model.append(layers.Activation(activation))
                    sequential_model.append(layers.Dropout(0.5))
                elif layer_type == 'LAST_DENSE':
                    sequential_model.append(layers.Dense(Globals.NUM_CLASSES, activation='softmax'))
                else:
                    print(f'ERROR: {layer_type} Layer type not recognized')
                    sequential_model = None
        return sequential_model

    def __init__(self, encoding = ConfigClass.ENCODING_TYPE, idx = 9999, genotypeObj = None):
        super().__init__(encoding, idx, genotypeObj)
        if is_None_or_empty(genotypeObj):
            self.integer_encoding = []
        else:
            self.integer_encoding = self.genList_to_integer_vector()
            self.integer_size = len(self.integer_encoding)
            if encoding == 'BIN':
                self.integer_encoding_to_binary_encoding(Globals.INDEXES_DENSES[-1])
