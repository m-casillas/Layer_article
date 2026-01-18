from globalsENAS import *
from BlockRepresentation import BlockRepresentation
"""# Mutator classs"""

#This class mutates each layer randomly. CONV, POOLMAX, DENSE, etc
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class Mutator:

    def mutate_bitflip(self, binary_encoding, MAX_INT):
        #MBFLIP
        #Takes the binary encoding of the architecture. Then, it flips bits randomly.
        #Returns the modified binary encoding.
        #If MAX_INT (15) is surpassed, it reverses the last mutation.
        
        #[0, 8, 2, 8, 10]
        #['0000', '1100', '0011', '1100', '1111']
        #['0000', '1000', '0011', '1100', '1111']
        #[0, 15, 2, 8, 10]
        mut_binary_encoding = []
        for bitstring in binary_encoding:
            new_bitstr = bitstring
            for i in range(len(bitstring)):
                if random.random() < 1/len(binary_encoding):
                    beforemut_bitstr = new_bitstr #Old version of new_bitstr (before mutation)
                    new_bitstr = bitflip(new_bitstr, i)    
                    if gray_to_int(new_bitstr) > MAX_INT:
                        new_bitstr = beforemut_bitstr
            mut_binary_encoding.append(new_bitstr)
        return mut_binary_encoding
    
    def mutate_block_parameters(self, gen_obj, block_idx):
        #Change the actual block to another block (the next one in order), from all_blocks.
        #block_idx is the index in gen_list where the block is located but I need to get its index in all_blocks
        idxBlock = BlockRepresentation.get_block_index(gen_obj.gen_list[block_idx])
        new_block_idx = idxBlock + 1
        new_block_idx  = check_within_bounds(new_block_idx, 0, len(BlockRepresentation.all_blocks)-1)
        gen_obj.gen_list[block_idx] = BlockRepresentation.all_blocks[new_block_idx]
        return gen_obj.gen_list
    
    def mutate_block_swap(self, gen_obj, block_idx):
        first_block_idx = block_idx
        second_block_idx = random_choice_except(BLOCKS_CONSTANTS.MUTABLE_BSWAP_INDEXES, first_block_idx)
        #Swap the two blocks
        gen_obj.gen_list[first_block_idx], gen_obj.gen_list[second_block_idx] = gen_obj.gen_list[second_block_idx], gen_obj.gen_list[first_block_idx]
        return gen_obj.gen_list

    def mutate_layer_type(self, gen_obj, layer_idx):
        #'MTYPE'
        #This function changes layers types (CONV to POOL, or POOL to CONV)
        
        layer_to_mutate = gen_obj.gen_list[layer_idx]
        layer_type = list(layer_to_mutate.keys())[0]
        new_layer_type = random_choice_except(Globals.type_mutable_layers, layer_type)
        create_layer_func = Globals.create_layers_functions_dict[new_layer_type]
        gen_obj.gen_list[layer_idx] = create_layer_func()
        return gen_obj.gen_list

    def mutate_layer_parameters(self, gen_obj, layer_idx): 
        #Adapt this for more representations
        #'MPAR'
        #Identifies the current layer type and changes its parameters to the next ones in order.
        layer_to_mutate = gen_obj.gen_list[layer_idx]
        layer_type = list(layer_to_mutate.keys())[0]
        idxLayer = Globals.all_layers.index(layer_to_mutate)
        new_layer_idx = idxLayer + 1        
        if layer_type == 'CONV':
            new_layer_idx = check_within_bounds(new_layer_idx, Globals.INDEXES_CONVS[0], Globals.INDEXES_CONVS[-1])
            gen_obj.gen_list[layer_idx] = Globals.all_layers[new_layer_idx]
        
        elif layer_type in ['POOLMAX', 'POOLAVG'] :
            new_layer_idx = check_within_bounds(new_layer_idx, Globals.INDEXES_POOLS[0], Globals.INDEXES_POOLS[-1])
            gen_obj.gen_list[layer_idx] = Globals.all_layers[new_layer_idx]
        elif layer_type == 'DENSE':
            new_layer_idx = check_within_bounds(new_layer_idx, Globals.INDEXES_DENSES[0], Globals.INDEXES_DENSES[-1])
            gen_obj.gen_list[layer_idx] = Globals.all_layers[new_layer_idx]
        else:
            print(f'ERROR: {layer_type} Layer type not recognized (mutate_later_parameters)')
                
        return gen_obj.gen_list

    def __init__(self):
        self.arch_obj = None
        
