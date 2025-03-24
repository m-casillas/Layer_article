from globalsENAS import *
from utilitiesENAS import *
"""# Mutator classs"""

#This class mutates each layer randomly. CONV, POOLMAX, DENSE, etc
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class Mutator:
    def mutate_layer_type(self, gen_obj, layer_idx):
        #'L_MODIFY_PARAMS'

        #This function changes layers types (CONV to POOL, or POOL to CONV)
        #Returns the new layer that will replace the current one and also the layer type, if it were needed
        self.layer_idx = layer_idx
        layer_mutated = gen_obj.gen_list[self.layer_idx]
        layer_type = list(layer_mutated.keys())[0]
        temp_genotype = gen_obj #Remember, genotype objects have an attribute "gen_list": a list of dictionaries. Each dictionary is a layer
        
        #Remember, create_pool_layer returns the dictionary.
        if layer_type == 'CONV':
            #Change it to a POOLMAX layer
            temp_genotype.gen_list[self.layer_idx] = create_pool_layer()
            layer_type = 'POOLMAX'
        elif layer_type == 'POOLMAX':
            #Change it to a CONV layer
            temp_genotype.gen_list[self.layer_idx] = create_conv_layer()
            layer_type = 'CONV'
        else:
            print(f'ERROR: {layer_type} Layer type not recognized')
            temp_genotype.gen_list[self.layer_idx] = None
        
        return temp_genotype.gen_list[self.layer_idx], layer_type

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
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_conv_layer()[layer_type]
        elif layer_type == 'POOLMAX':
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_pool_layer()[layer_type]
        elif layer_type == 'DENSE':
            temp_genotype.gen_list[self.layer_idx][layer_type] = create_dense_layer()[layer_type]
        else:
            print(f'ERROR: {layer_type} Layer type not recognized')
            temp_genotype.gen_list[self.layer_idx][layer_type] = None
        
        return temp_genotype.gen_list[self.layer_idx], layer_type
    def __init__(self, ):
        #unit
        self.arch_obj = None
        self.layer_idx = 9999