from globalsENAS import *
import numpy as np
"""# Mutator classs"""

#This class mutates each layer randomly. CONV, POOLMAX, DENSE, etc
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class Mutator:
    def mutate_layer_type(self, gen_obj, layer_idx):
        #'L_MODIFY_TYPE'

        #This function changes layers types (CONV to POOL, or POOL to CONV)
        #Returns the new layer that will replace the current one and also the layer type, if it were needed
        self.layer_idx = layer_idx
        layer_mutated = gen_obj.gen_list[self.layer_idx]
        layer_type = list(layer_mutated.keys())[0]
        temp_genotype = gen_obj #Remember, genotype objects have an attribute "gen_list": a list of dictionaries. Each dictionary is a layer
        new_layer_type = select_type_filtering(type_mutable_layers, layer_type)
        create_layer_func = create_layers_functions_dict[new_layer_type]
        #Remember, create_pool_layer returns the dictionary.

        temp_genotype.gen_list[self.layer_idx] = create_layer_func()
        return temp_genotype.gen_list[self.layer_idx], new_layer_type

    def mutate_layer_parameters(self, gen_obj, layer_idx): #Adapt this for more representations
        #'L_MODIFY_PARAMS'
        #Returns the specific layer that mutated (the dictionary)
        #Returns the new layer that will replace the current one and also the layer type, if it were needed

        self.layer_idx = layer_idx
        layer_to_mutate = gen_obj.gen_list[self.layer_idx]
        layer_type = list(layer_to_mutate.keys())[0]
        temp_genotype = gen_obj #Remember, genotype objects have an attribute "gen_list": a list of dictionaries. Each dictionary is a layer
        #Remember, create_pool_layer returns the dictionary. You only need the value, not the key "layer_type"
        
        if layer_type == 'CONV':
            #For CONV, randomly choose between changing the number of filters or the kernel size.
            nf = layer_to_mutate['CONV'][0]
            ks = layer_to_mutate['CONV'][1]
            if np.random.rand() < 0.5:
                nf += 1
                nf = add_within_bounds(nf, CONV_MINFILTER_IND, CONV_MAXFILTER_IND)
            else:
                ks += 1
                ks = add_within_bounds(ks, CONV_MINKERNEL_IND, CONV_MAXKERNEL_IND)
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_conv_layer(nf,ks)[layer_type]
        elif layer_type == 'POOLMAX':
            ks = layer_to_mutate['POOLMAX'][1]
            ks = add_within_bounds(ks, CONV_MINKERNEL_IND, CONV_MAXKERNEL_IND)
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_pool_max_layer(ks)[layer_type]
        elif layer_type == 'DENSE':
            nn = layer_to_mutate['DENSE'][0]
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_dense_layer(nn)[layer_type]
        else:
            print(f'ERROR: {layer_type} Layer type not recognized')
            temp_genotype.gen_list[self.layer_idx][layer_type] = None
        
        return temp_genotype.gen_list[self.layer_idx], layer_type
    def __init__(self):
        #unit
        self.arch_obj = None
        self.layer_idx = 9999

from Genotype import *
from LayerRepresentation import *
gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
gen = Genotype('L', 'I', gen_list)
arch = LayerRepresentation('S', 0, gen)
print(arch.genoStr)
mut = Mutator()
print(mut.mutate_layer_type(gen, 1))
