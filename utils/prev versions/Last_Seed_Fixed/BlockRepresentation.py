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
        
    def integer_encoding_to_real_encoding(self):
        self.real_encoding = []
        for layer_int in self.integer_encoding:
            layer_real = layer_int + random.random()
            self.real_encoding.append(layer_real)

    def real_encoding_to_integer_encoding(self):
        self.integer_encoding = []
        for layer_real in self.real_encoding:
            layer_int = int(layer_real)
            self.integer_encoding.append(layer_int)
    
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

    def integer_vector_to_genList(self, integer_vector):
        #Given an integer vector, returns the gen list (list of dictionaries). The input, the GLOBAL and the last Dense layer are not included in the integer vector.
        gen_list = []
        block_count = 0
        for chromosome in self.genotype.gen_list:
            if isinstance(chromosome, dict): #it's a layer, not included in the integer vector.
                gen_list.append(chromosome)
            elif isinstance(chromosome, tuple): #it's a block
                block_idx = integer_vector[block_count]
                #print(f'Block index: {block_idx}')
                block = BlockRepresentation.all_blocks[block_idx]
                gen_list.append(block)
                block_count += 1
            else:
                print('integer_vector_to_genList: Unidentified chromosome.')
                gen_list.append(None)
                                
        return gen_list

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
        


    #def __init__(self, encoding_type = ConfigClass.ENCODING_TYPE, idx = 9999, genotypeObj = None):
    #    super().__init__(encoding = encoding_type, idx = idx, genotypeObj = genotypeObj)

    def __init__(self, encoding = ConfigClass.ENCODING_TYPE, idx = 9999, genotypeObj = None):
        super().__init__(encoding, idx, genotypeObj)
        if is_None_or_empty(genotypeObj):
            self.integer_encoding = []
        else:
            self.integer_encoding = self.genList_to_integer_vector()
            self.integer_size = len(self.integer_encoding)
            if encoding == 'BIN':
                self.integer_encoding_to_binary_encoding(len(BlockRepresentation.all_blocks)-1)
            elif encoding == 'REAL':
                self.integer_encoding_to_real_encoding()
            else:
                print(f'ERROR: Encoding type {encoding} not recognized')
                self.integer_encoding = []



