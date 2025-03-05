from globalsENAS import *
from Architecture import *
"""#Layer Representation Class"""

#Architecture is a deep neural network consisting in the input, conv layer, pooling layer, conv layer, pooling layer, fully connected layer
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class LayerRepresentation(Architecture):
   
    #Creates the architecture using the genotype information
    def decode(self):
        if self.genotype.rep_type == 'L':
            #This list stores each layer to create the Sequential model
            sequential_model = []
            for layer in self.genotype.gen_list:
                layer_type = list(layer.keys())[0] #Get the key of the dictionary "the layer type"
                if layer_type == 'INP':
                    input_size = layer[layer_type]
                    sequential_model.append(Input(shape=(input_size, input_size, 1)))
                elif layer_type == 'CONV':
                    num_filters = layer[layer_type][0]
                    kernel_size = (layer[layer_type][1], layer[layer_type][1])
                    sequential_model.append(layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same'))
                elif layer_type == 'POOLMAX':
                    pool_size = (layer[layer_type], layer[layer_type])
                    sequential_model.append(layers.MaxPooling2D(pool_size))
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