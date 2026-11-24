from globalsENAS import *
from BlockRepresentation import BlockRepresentation
"""# Mutator classs"""

#This class mutates each layer randomly. CONV, POOLMAX, DENSE, etc
#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
class Mutator:

    def mutate_bitflip(self, binary_encoding = None, MAX_INT = 0, INTEGER_ENCODING2 = False, MUT_PROB = 0.1):
        # MBFLIP
        # Bit-flip mutation for variable-length Gray-coded genes.
        # Even positions and odd positions can have different MAXINT.
        # If a mutation exceeds the allowed range, it is reversed.
        mut_binary_encoding = []
        for i, bitstring in enumerate(binary_encoding):
            new_bitstr = bitstring
            # Obtain the maximum integer allowed for each gene
            if INTEGER_ENCODING2:
                LOWINT, MAXINT = low_high_integer2(i)
            else:
                LOWINT, MAXINT = 0, MAX_INT
                     
            for j in range(len(bitstring)):
                if random.random() < MUT_PROB:
                    beforemut_bitstr = new_bitstr
                    # Flip the selected bit
                    new_bitstr = bitflip(new_bitstr, j)
                    # Validate the decoded Gray value
                    decoded_value = gray_to_int(new_bitstr)
                    if not (LOWINT <= decoded_value <= MAXINT):
                        new_bitstr = beforemut_bitstr
            mut_binary_encoding.append(new_bitstr)
        return mut_binary_encoding

    def mutate_bitflip2(self, binary_encoding = None, MAX_INT = 0, INTEGER_ENCODING2 = False):
        #MBFLIP
        #Takes the binary encoding of the architecture. Then, it flips bits randomly.
        #Returns the modified binary encoding.
        #If MAX_INT (15) is exceeded, it reverses the last mutation.
        
        #[0, 8, 2, 8, 10]
        #['0000', '1100', '0011', '1100', '1111']
        #['0000', '1000', '0011', '1100', '1111']
        #[0, 15, 2, 8, 10]
        mut_binary_encoding = []
        for i, bitstring in enumerate(binary_encoding):
            new_bitstr = bitstring
            LOWINT, MAXINT = low_high_integer2(i) if INTEGER_ENCODING2 else (0, MAX_INT)
            for j in range(len(bitstring)):
                if random.random() < 1/len(binary_encoding):
                    beforemut_bitstr = new_bitstr
                    new_bitstr = bitflip(new_bitstr, j)
                    if gray_to_int(new_bitstr) > MAX_INT:
                        new_bitstr = beforemut_bitstr
            mut_binary_encoding.append(new_bitstr)
        return mut_binary_encoding

     
    

    def mutate_layer_swap(self, gen_obj, layer_idx):
        
        # Swap two randomly chosen genes within the mutable CONV/POOL zone.
        # Excludes index 1 (fixed first CONV) and index 6 (mandatory last POOL).
        idx1 = layer_idx
        idx2 = random_choice_except(LAYERS_CONSTANTS.MUTABLE_LSWAP_INDEXES, idx1)
        gen_obj.gen_list[idx1], gen_obj.gen_list[idx2] = gen_obj.gen_list[idx2], gen_obj.gen_list[idx1]
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



    def mutate_block_parameters(self, vec_int = None, INTEGER_ENCODING2 = False, MUT_PROB = 0.1):
            #MPAR
            #Change the actual block to another block (the next one in order), from all_blocks.
            #block_idx is the index in gen_list where the block is located but I need to get its index in all_blocks
            for i in range(len(vec_int)):
                MIN_INT, MAX_INT = low_high_integer2(i) if INTEGER_ENCODING2 else (0, len(BlockRepresentation.all_blocks)-1)
                if random.random() < MUT_PROB:
                    vec_int[i] += 1
                    vec_int[i] = check_within_bounds(vec_int[i], MIN_INT, MAX_INT)
            return vec_int

    def mutate_block_swap(self, vec_int = None, INTEGER_ENCODING2 = False, MUT_PROB = 0.1):
            #MSWAP
            increment = 2 if INTEGER_ENCODING2 else 1 #Because each two integers represent one block in INT2.
            mutable_indexes = range(0, len(vec_int), increment) if INTEGER_ENCODING2 else range(len(vec_int))
            for i in mutable_indexes:
                if random.random() < MUT_PROB:
                    first_idx = i
                    second_idx = random_choice_except(mutable_indexes, first_idx)
                    vec_int[first_idx], vec_int[second_idx] = vec_int[second_idx], vec_int[first_idx]
                    if INTEGER_ENCODING2:
                        vec_int[first_idx + 1], vec_int[second_idx + 1] = vec_int[second_idx + 1], vec_int[first_idx + 1]
            return vec_int
    
    def polynomial_mutation(self, vreal = None, t = None, ylow = None, yhigh = None, INTEGER_ENCODING2 = False, MUT_PROB = 0.1):
        #MPOLY
        nm = 100 + t
        mut_real_encoding = []
        for i, y in enumerate(vreal):
            if random.random() >= MUT_PROB:
                mut_real_encoding.append(y)
                continue

            u = random.random()
            add = nm + 1
            power = 1/add
            ylow, yhigh = low_high_integer2(i) if INTEGER_ENCODING2 else (ylow, yhigh)
            max_delta = yhigh - ylow
            delta = min(y-ylow, yhigh-y)/max_delta
            if u <= 0.5:
                delta = 2*u + (1-2*u)*(1-delta)**add
                delta = delta**power - 1
            else:
                delta = 2*(1-u) + 2*(u-0.5)*(1-delta)**add
                delta = 1 - delta**power
            mut_gene = y + delta*max_delta
            mut_gene = check_within_bounds(mut_gene, ylow, yhigh)
            mut_real_encoding.append(mut_gene)
        return mut_real_encoding

    def nouniform_mutation(self, vreal = None, t = None, ylow = None, yhigh = None, b=2, INTEGER_ENCODING2=False, MUT_PROB = 0.1):
        #MNUF
        def delta(t, y, r, T, b):
            return y * (1 - r ** ((1 - t / T) ** b))
        mut_real_encoding = []
        for i, y in enumerate(vreal):
            if random.random() >= MUT_PROB:
                mut_real_encoding.append(y)
                continue
            tau = random.randint(0, 1)
            r = random.random()       
            ylow, yhigh = low_high_integer2(i) if INTEGER_ENCODING2 else (ylow, yhigh)
            if tau == 0:
                mut_gene = y + delta(t, yhigh - y, r, config_tecnas.GENERATIONS, b)
            else:
                mut_gene = y - delta(t, y - ylow, r, config_tecnas.GENERATIONS, b)
            mut_gene = check_within_bounds(mut_gene, ylow, yhigh)
            mut_real_encoding.append(mut_gene)
        return mut_real_encoding

    def __init__(self):
        ...
        
