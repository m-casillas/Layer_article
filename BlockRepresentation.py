from LayerRepresentation import LayerRepresentation
from Architecture import *
from globalsENAS import *

class BlockRepresentation(Architecture):
    
    @staticmethod
    def create_list_blocks(NCONV_PERBLOCK = ConfigBlocks.NCONV_PERBLOCK):
        #Create all combinations of N CONVs, that is, one block
        all_blocks = []
        for combo in itertools.product(Globals.all_convs, repeat = NCONV_PERBLOCK):
            all_blocks.append(combo)
        print(f'Number of possible blocks ({NCONV_PERBLOCK} CONV per block): {len(all_blocks)}')
        #First block is  ({'CONV': [32, 3]}, {'CONV': [32, 3]})
        return all_blocks #List of tuples of N dictionaries
    
    all_blocks = create_list_blocks()
    @staticmethod
    def reset_add_pools(gen_list, reset = True):
        #Add a POOLMAX aftet two blocks.
        #If reset is true, removes all POOLMAX operations first.
        if reset:
            for i, block in enumerate(gen_list):
                if isinstance(block, tuple):
                    #It's a block (tuple of N dictionaries).  A block is uniquely identified by its index in the list of all blocks
                    if len(block) > ConfigBlocks.NCONV_PERBLOCK:
                        gen_list[i] = block[:ConfigBlocks.NCONV_PERBLOCK]
        for i, block in enumerate(gen_list):
            if i%2 != 0: #Each two blocks add a POOLMAX
                if isinstance(block, tuple):
                    block = block + ({'POOLMAX':[-1,2]},)
                    gen_list[i] = block

        #Validate number of POOLS
        pool_count = 0
        for block in gen_list:
            if isinstance(block, tuple):
                if len(block) > ConfigBlocks.NCONV_PERBLOCK:
                    pool_count += 1
        if pool_count != 4:
            print(f'Error in reset_add_pools: Number of POOLMAX is {pool_count}, should be 4.')
            return None
        return gen_list
    @staticmethod       
    def get_block_index(block):
        #Given a block (tuple of N dictionaries), returns its index in all_blocks
        #Some blocks have a POOL operation at the end, which is not part of the block.
        if len(block) > ConfigBlocks.NCONV_PERBLOCK:
            return BlockRepresentation.all_blocks.index(block[:ConfigBlocks.NCONV_PERBLOCK])
        else:
            return BlockRepresentation.all_blocks.index(block)

    @staticmethod
    def integer_encoding_to_integer_encoding2(integer_encoding):
        #If integer_encoding is [0, 2, 55] the integer_encoding2 is [ (0,0), (0,0), (0,0), (1,0), (3,0), (3,1)]
        #Each integer represents two CONVS, and each CONV has filters and kernels. That's why each integer results in two tuples each.
        #CONV_KERNEL_LIST =   [3,   5]
        #NUM_FILTERS_LIST =   [32, 64,  128, 256]
        temp_integer_encoding = []
        for layer_int in integer_encoding:
            block = BlockRepresentation.all_blocks[layer_int] #({'CONV': [32, 3]}, {'CONV': [32, 5]})    for example, if layer_int is 1
            temp_block_list = [] #Here i'll store each single CONV, such as [0,0] and then cast it to a tuple.
            for conv in block: #conv is {'CONV': [32, 3]}  then {'CONV': [32, 5]}
                filt   = conv['CONV'][0] #This is 32
                kernel = conv['CONV'][1] #This is 3
                filt_idx   = Globals.NUM_FILTERS_LIST.index(filt) #This is 0
                kernel_idx = Globals.CONV_KERNEL_LIST.index(kernel) #This is 0
                temp_block_list.append((filt_idx, kernel_idx)) #This is (0,0)
            temp_integer_encoding.extend(temp_block_list) #This is [(0,0), (0,1)] for example
        #self.integer_encoding = 
        return temp_integer_encoding

    @staticmethod
    def integer_encoding2_to_integer_encoding(integer_encoding2):
        #If integer_encoding2 is [(0,0), (0,0), (0,0), (1,0), (3,0), (3,1)] the integer_encoding is [0, 2, 55]
        temp_integer_encoding = []
        cont_convs = 0 #Each 2 CONVs are a block.
        single_block = []
        for single_conv in integer_encoding2:
            filt_idx = single_conv[0]
            kernel_idx = single_conv[1]
            filt_num = Globals.NUM_FILTERS_LIST[filt_idx]
            kernel_num = Globals.CONV_KERNEL_LIST[kernel_idx]
            CONV_dict = {'CONV':[filt_num, kernel_num]}
            single_block.append(CONV_dict) # [{'CONV': [32, 5]}]
            cont_convs += 1
            if cont_convs == ConfigBlocks.NCONV_PERBLOCK:
                #single block: [{'CONV': [32, 3]},{'CONV': [32, 5]}]
                block_index = BlockRepresentation.get_block_index(tuple(single_block)) #This is the index of the block in all_blocks
                temp_integer_encoding.append(block_index)
                cont_convs = 0
                single_block = []
        return  temp_integer_encoding

    @staticmethod
    def flattened_integer_encoding2_to_integer_encoding(flattened_integer_encoding2):
        #If flattened_integer_encoding2 is [0,0,0,0,0,0,1,0,3,0,3,1] the integer_encoding is [0, 2, 55]
        integer_encoding2 = BlockRepresentation.flattened_to_tuples(flattened_integer_encoding2)
        integer_encoding = BlockRepresentation.integer_encoding2_to_integer_encoding(integer_encoding2)
        return integer_encoding

    @staticmethod
    def tuples_to_flattened(list_tuples):
        #If list_tuples is [(0,0), (0,0), (0,0), (1,0), (3,0), (3,1)] the flattened version is [0,0,0,0,0,0,1,0,3,0,3,1]
        return  [x for tupla in list_tuples for x in tupla]

    @staticmethod
    def flattened_to_tuples(flattened_vector):
            #If flattened_vector_encoding2 is [0,0,0,0,0,0,1,0,3,0,3,1] the deflattened_vector_encoding2 is [(0,0), (0,0), (0,0), (1,0), (3,0), (3,1)]
            #Works also with real encodings.
            if len(flattened_vector) % 2 != 0:
                print(f'Error in deflatten_vector_encoding2: Length of flattened_vector_encoding2 is {len(flattened_vector)}, should be even.')
                return None
            return [(flattened_vector[i], flattened_vector[i+1]) for i in range(0, len(flattened_vector), 2)]

    @staticmethod
    def integer_encoding_to_real_encoding(integer_encoding):
        #if integer_encoding is [0, 2, 55] the real_encoding may be [0.1234, 2.5678, 55.91011]  (random numbers added to each integer)
        real_encoding = []
        for layer_int in integer_encoding:
            layer_real = layer_int + random.random()
            real_encoding.append(layer_real)
        return real_encoding

    @staticmethod
    def real_encoding_to_integer_encoding(real_encoding):
        #If real_encoding is [0.1234, 2.5678, 55.91011] the integer_encoding is [0, 2, 55]
        integer_encoding = []
        for layer_real in real_encoding:
            layer_int = int(layer_real)
            integer_encoding.append(layer_int)
        return integer_encoding

    @staticmethod
    def flattened_real_encoding2_to_integer_encoding2(flattened_real_encoding2):
        #If flattened_real_encoding2 is [0.1, 0.2, 5.4, 20.5] the integer_encoding2 is [(0, 5), (0, 20)]
        real_encoding2 = BlockRepresentation.flattened_to_tuples(flattened_real_encoding2)
        integer_encoding2 = BlockRepresentation.real_encoding2_to_integer_encoding2(real_encoding2)
        return integer_encoding2

    @staticmethod
    def real_encoding2_to_integer_encoding2(real_encoding2):
        #If real_encoding2 is [(0.1, 5.4), (0.2, 20.5)], integer_encoding2 is [(0, 5), (0, 20)]
        integer_encoding2 = []
        for layer_real_tuple in real_encoding2:
            integer_encoding2.append((int(layer_real_tuple[0]), int(layer_real_tuple[1])))
        return integer_encoding2

    @staticmethod
    def integer_encoding2_to_real_encoding2(integer_encoding2) :
        #If integer_encoding2 is [(0, 5), (0, 20)], real_encoding2 is [(0.1, 5.4), (0.2, 20.5)] (random numbers added to each integer)
        real_encoding2 = []
        for layer_int_tuple in integer_encoding2:
            real_encoding2.append((layer_int_tuple[0] + random.random(), layer_int_tuple[1] + random.random()))
        return real_encoding2

    @staticmethod
    def flattened_real_encoding2_to_integer_encoding(flattened_real_encoding2):
        #Go from flattened real encoding2 to real encoding2 to integer encoding2 to integer encoding
        real_encoding2 = BlockRepresentation.flattened_to_tuples(flattened_real_encoding2)
        integer_encoding2 = BlockRepresentation.real_encoding2_to_integer_encoding2(real_encoding2)
        integer_encoding = BlockRepresentation.integer_encoding2_to_integer_encoding(integer_encoding2)
        return integer_encoding
        
    @staticmethod #use INTEGER_ENCODING2 Flag
    def integer_encoding_to_binary_encoding(integer_encoding = None, maxInt = None, INTEGER_ENCODING2 = False):
        binary_encoding = []
        for i,layer_int in enumerate(integer_encoding):
            maxInt = low_high_integer2(i)[1] if INTEGER_ENCODING2 else maxInt
            #print(f'BlockRepresentation.integer_encoding_to_binary_encoding: layer_int = {layer_int}, maxInt = {maxInt}')
            layer_bin = int_to_gray(layer_int, maxInt)
            binary_encoding.append(layer_bin)
        return binary_encoding
    @staticmethod
    def binary_encoding_to_integer_encoding(binary_encoding = None):
        integer_encoding = []
        for layer_bin in binary_encoding:
            layer_int = gray_to_int(layer_bin)
            integer_encoding.append(layer_int)
        return integer_encoding

    def genList_to_integer_vector(self):
        #Represents the gen list (list of dictionaries) as an integer vector. The input and the last Dense layer are not included.
        integer_vector = []
        #chromosome can be a dictionary (single operation like CONV, POOL) or a tuple of dictionaries (a block)
        for chromosome in self.genotype.gen_list:
            if isinstance(chromosome, dict): #it's a layer, not included in the integer vector.
                continue
            elif isinstance(chromosome, tuple): #it's a block
                #It's a block (tuple of N dictionaries).  A block is uniquely identified by its index in the list of all blocks
                block_idx = BlockRepresentation.get_block_index(chromosome)
                integer_vector.append(block_idx)  
            else:
                print('genList_to_integer_vector: Unidentified chromosome.')
                integer_vector.append(None)
                                
        return integer_vector

    @staticmethod
    def integer_vector_to_genList(integer_vector = None):
        gen_list = []
        for intnum in range(len(integer_vector)): #integer_vector =[32, 4, 34, 43, 51, 61, 39, 0]
            block_idx = integer_vector[intnum]
            gen_list.append(BlockRepresentation.all_blocks[block_idx])

        gen_list = BlockRepresentation.update_genlist(gen_list)
        return gen_list

    @staticmethod
    def update_genlist(gen_list):
        #After crossover and mutation, input and globalavg and dense layers are missing. Add them back.
        gen_list.insert(0, {'INP':Globals.INPUT_SIZE})
        gen_list.insert(1, create_conv_layer(nf = 64, ks = 3))
        gen_list.insert(len(gen_list),create_globalAVG_layer())
        gen_list.insert(len(gen_list),create_dense_layer(last_dense = True))
        gen_list = BlockRepresentation.reset_add_pools(gen_list, reset = True)
        
        #print()
        return gen_list
    @staticmethod
    def update_vector_encodings(int_vec = None, INTEGER_ENCODING2 = False, binary_encoding = False):
        real_encoding = BlockRepresentation.integer_encoding_to_real_encoding(int_vec)
        integer_encoding2 = BlockRepresentation.integer_encoding_to_integer_encoding2(int_vec) if INTEGER_ENCODING2 else []
        integer_encoding2_flattened = BlockRepresentation.tuples_to_flattened(integer_encoding2) if INTEGER_ENCODING2 else []
        real_encoding2 = BlockRepresentation.integer_encoding2_to_real_encoding2(integer_encoding2) if INTEGER_ENCODING2 else []
        real_encoding2_flattened = BlockRepresentation.tuples_to_flattened(real_encoding2) if INTEGER_ENCODING2 else []
        binary_encoding = BlockRepresentation.integer_encoding_to_binary_encoding(int_vec, maxInt = len(BlockRepresentation.all_blocks)-1, INTEGER_ENCODING2 = INTEGER_ENCODING2) if binary_encoding else []
        binary_encoding2_flattened = BlockRepresentation.integer_encoding_to_binary_encoding(integer_encoding = integer_encoding2_flattened, maxInt = len(BlockRepresentation.all_blocks)-1, INTEGER_ENCODING2 = INTEGER_ENCODING2) if INTEGER_ENCODING2 else []
        return int_vec, integer_encoding2, integer_encoding2_flattened, real_encoding, real_encoding2, real_encoding2_flattened, binary_encoding, binary_encoding2_flattened

    
    def residual_block(self, x, list_filters, list_kernels, pool = False, stride=1, training=True):
        shortcut = x
        for filter, kernel in zip(list_filters[:-1], list_kernels[:-1]):
            x = layers.Conv2D(filter, kernel_size = kernel, strides=stride, padding='same', use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4))(x)  # L2 regularization
            x = layers.BatchNormalization()(x, training=training)
            x = layers.ReLU()(x)
        #LAST CONV WIT NO RELU YET
        x = layers.Conv2D(list_filters[-1], kernel_size = list_kernels[-1], strides=1, padding='same', use_bias=False,
                        kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x, training=training)
        if stride != 1 or shortcut.shape[-1] != list_filters[:-1]:
            shortcut = layers.Conv2D(list_filters[-1], 1, strides=stride, padding='same', use_bias=False,
                                    kernel_regularizer=regularizers.l2(1e-4))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut, training=training)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        if pool:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        return x
    
    def decode(self):
        #Create the keras model from the gen_list
        inputs = layers.Input(shape = (Globals.INPUT_SIZE, Globals.INPUT_SIZE, 3))
        #First CONV is not part of the blocks. Get the filters and kernel from this layer.
        f = self.genotype.gen_list[1]['CONV'][0]
        k = self.genotype.gen_list[1]['CONV'][1]
        x = layers.Conv2D(f, k, strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        for block in self.genotype.gen_list[2:-2]: #From the first block, before the Global Pooling
            filter_list = []
            kernel_list = []
            for conv in block[:ConfigBlocks.NCONV_PERBLOCK]: #Ignore the POOL if it exists
                filter_list.append(conv['CONV'][0])
                kernel_list.append(conv['CONV'][1])
            add_pool = len(block) > ConfigBlocks.NCONV_PERBLOCK
            x = self.residual_block(x, filter_list, kernel_list, pool = add_pool, stride=1, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)  # Adjusted to 0.4 for balance
        outputs = layers.Dense(Globals.NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        #model.summary()
        return model
        


    def __init__(self, encoding = ConfigClass.ENCODING_TYPE, idx = 9999, genotypeObj = None, INTEGER_ENCODING2 = False):
        super().__init__(encoding, idx, genotypeObj)
        self.INTEGER_ENCODING2 = INTEGER_ENCODING2
        if is_None_or_empty(genotypeObj):
            self.integer_encoding = []
        else:
            self.integer_encoding = self.genList_to_integer_vector()
            self.integer_size = len(self.integer_encoding)
            self.integer_encoding2 = BlockRepresentation.integer_encoding_to_integer_encoding2(self.integer_encoding) if INTEGER_ENCODING2 else []
            self.integer_encoding2_flattened = BlockRepresentation.tuples_to_flattened(self.integer_encoding2) if INTEGER_ENCODING2 else []
            if encoding == 'BIN':               
                self.binary_encoding = self.integer_encoding_to_binary_encoding(integer_encoding = self.integer_encoding, maxInt = len(BlockRepresentation.all_blocks)-1, INTEGER_ENCODING2 = INTEGER_ENCODING2)
                self.binary_encoding2_flattened = self.integer_encoding_to_binary_encoding(integer_encoding = self.integer_encoding2_flattened, maxInt = len(BlockRepresentation.all_blocks)-1, INTEGER_ENCODING2 = INTEGER_ENCODING2) if INTEGER_ENCODING2 else []
                
            elif encoding == 'REAL':
                self.real_encoding = BlockRepresentation.integer_encoding_to_real_encoding(self.integer_encoding)
                self.real_encoding2 = BlockRepresentation.integer_encoding2_to_real_encoding2(self.integer_encoding2) if INTEGER_ENCODING2 else []
                self.real_encoding2_flattened = BlockRepresentation.tuples_to_flattened(self.real_encoding2) if INTEGER_ENCODING2 else []
            elif encoding == 'INT':
                ...
                #self.integer_encoding2 = BlockRepresentation.integer_encoding_to_integer_encoding2(self.integer_encoding)
                #self.integer_encoding2_flattened = BlockRepresentation.tuples_to_flattened(self.integer_encoding2)
                
            else:
                print(f'ERROR: Encoding type {encoding} not recognized')
                self.integer_encoding = []


'''os.system("cls")
from Genotype import Genotype
gen_list = [{'INP': 32}, {'CONV': [64, 3]}, ({'CONV': [32, 3]}, {'CONV': [32, 5]}), ({'CONV': [64, 5]}, {'CONV': [64, 5]}, {'POOLMAX': [-1, 2]}), ({'CONV': [128, 3]}, {'CONV': [256, 3]}), ({'CONV': [128, 3]}, {'CONV': [256, 3]}, {'POOLMAX': [-1, 2]}), ({'CONV': [256, 5]}, {'CONV': [32, 5]}), ({'CONV': [64, 5]}, {'CONV': [256, 3]}, {'POOLMAX': [-1, 2]}), ({'CONV': [32, 5]}, {'CONV': [64, 3]}), ({'CONV': [128, 5]}, {'CONV': [64, 3]}, {'POOLMAX': [-1, 2]}), {'GLOBAL_AVG': None}, {'LAST_DENSE': [100, 'softmax']}]
genotypeObj = Genotype(rep_type = ConfigClass.REPRESENTATION_TYPE, encoding_type = ConfigClass.ENCODING_TYPE, gen_list = gen_list)
blockObj = BlockRepresentation(encoding = ConfigClass.ENCODING_TYPE, idx = 9999, genotypeObj = genotypeObj, INTEGER_ENCODING2 = True)
print(f'Integer encoding2_flattened: {blockObj.integer_encoding2_flattened}')
print(f'Binary encoding2 flattened: {blockObj.binary_encoding2_flattened}')
print()
'''