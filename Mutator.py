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

    def mutate_layer_parameters(self, gen_obj, layer_idx): 
        #Adapt this for more representations
        #'L_MODIFY_PARAMS'
        #Returns the specific layer that mutated (the dictionary)
        #Returns the new layer that will replace the current one and also the layer type, if it were needed
        self.layer_idx = layer_idx
        layer_to_mutate = gen_obj.gen_list[self.layer_idx]
        layer_type = list(layer_to_mutate.keys())[0]
        temp_genotype = gen_obj #Remember, genotype objects have an attribute "gen_list": a list of dictionaries. Each dictionary is a layer
        #Remember, create_pool_layer returns the dictionary. You only need the value, not the key "layer_type"
        #layer_to_mutate is like {'POOLMAX': [-1, 2]}
        #layer_type is 'POOLMAX'
        
        nf = layer_to_mutate[layer_type][0]
        ks = layer_to_mutate[layer_type][1]
        
        if layer_type == 'CONV':
            nfidx = get_key_from_value(NUM_FILTERS, nf)
            ksidx = get_key_from_value(CONV_KERNELS, ks)
            #print(f'{nf = } {ks = } {nfidx = } {ksidx = }')
            if np.random.rand() < 0.5:
                nfidx += 1
                nfidx = check_within_bounds(nfidx, CONV_MINFILTER_IND, CONV_MAXFILTER_IND)
            else:
                ksidx += 1
                ksidx = check_within_bounds(ksidx, CONV_MINKERNEL_IND, CONV_MAXKERNEL_IND)
            nf = NUM_FILTERS[nfidx]
            ks = CONV_KERNELS[ksidx]
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_conv_layer(nf,ks)[layer_type]

        elif layer_type in ['POOLMAX', 'POOLAVG'] :
            ksidx = get_key_from_value(POOL_KERNELS, ks)
            #print(f'{ks = } {ksidx = }')
            ksidx += 1
            ksidx = check_within_bounds(ksidx, POOL_MINKERNEL_IND, POOL_MAXKERNEL_IND)
            ks = POOL_KERNELS[ksidx]
            if layer_type == 'POOLMAX':
                temp_genotype.gen_list[self.layer_idx][layer_type] = create_pool_max_layer(ks)[layer_type]
            else:
                temp_genotype.gen_list[self.layer_idx][layer_type] = create_pool_avg_layer(ks)[layer_type]
        elif layer_type == 'DENSE':
            nnidx = get_key_from_value(DENSE_NEURONS, nf)
            #print(f'{nf = } {nnidx = }')
            nnidx += 1 #nf is number of neurons, or nn
            nnidx = check_within_bounds(nnidx, DENSE_MINNEURONS_IND, DENSE_MAXNEURONS_IND)
            nn = DENSE_NEURONS[nnidx]
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_dense_layer(nn, 'relu')[layer_type]
        else:
            print(f'ERROR: {layer_type} Layer type not recognized (mutate_later_parameters)')
            temp_genotype.gen_list[self.layer_idx][layer_type] = None
        
        return temp_genotype.gen_list[self.layer_idx], layer_type

    def __init__(self):
        #unit
        self.arch_obj = None
        self.layer_idx = 9999



