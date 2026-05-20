'''
import psutil
import math
import tensorflow as tf
from colorama import Fore, Back, Style, init
init(autoreset=True)
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
#from tensorflow.python.profiler.model_analyzer import profile
#from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
from globalsENAS import *
from Surrogate_ENAS import Surrogate_ENAS
from Genotype import *
from ReportENAS import *
from LayerRepresentation import *
from BlockRepresentation import *
from Crossover import *
from Mutator import *
from pympler import asizeof
import tracemalloc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


"""# TECNAS Classs"""

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#genotype = Genotype('L', 'IV', gen_list)
#arch = Architecture('X', 0, 0, 0, 0, 1, genotype)

#TECNAS is a class that performs evoltuionary neural architecture search, using genetic operators
class TECNAS:
    if ConfigClass.REPRESENTATION_TYPE == 'L':
        REP_CONSTANTS = LAYERS_CONSTANTS
    elif ConfigClass.REPRESENTATION_TYPE == 'B':
        REP_CONSTANTS = BLOCKS_CONSTANTS
    def compute_confusion_matrix(self, arch_obj):
        y_pred_probs = arch_obj.model.predict(self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        arch_obj.cm = confusion_matrix(self.y_test, y_pred)
        
    def compute_metrics_from_cm(self, arch_obj):
        # Derive y_true and y_pred from the confusion matrix
        y_true = []
        y_pred = []
        num_classes = arch_obj.cm.shape[0]

        for true_label in range(num_classes):
            for pred_label in range(num_classes):
                count = arch_obj.cm[true_label, pred_label]
                y_true.extend([true_label] * count)
                y_pred.extend([pred_label] * count)

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        arch_obj.cm_accuracy = accuracy_score(y_true, y_pred)
        arch_obj.cm_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        arch_obj.cm_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        arch_obj.cm_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        '''
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        '''

    def flops_params_gen_list_str(self, gen_list_str):
        #Returns FLOPs and NumParams for a given gen_list
        memory = psutil.virtual_memory()
        print(Fore.RED + f"RAM used: {memory.percent}%")
        arch_obj = self.create_arch_from_genlist(ast.literal_eval(gen_list_str))
        flops =  calculate_model_flops(arch_obj.model)
        num_params = calculate_model_params(arch_obj.model)
        return flops, num_params

    def create_arch_from_genlist(self, gen_list):
        #Create an architecture from a gen_list
        arch_obj = LayerRepresentation('S', 'ARCH', genotypeObj=Genotype(gen_list=gen_list)) if ConfigClass.REPRESENTATION_TYPE == 'L' else BlockRepresentation('Sk', 'ARCH', genotypeObj=Genotype(gen_list=gen_list))
        arch_obj.model = self.create_model(arch_obj)
        return arch_obj

    def copy_arch2(self, arch1):
        #Copy the architecture from arch1 to arch2
        arch2 = LayerRepresentation() if ConfigClass.REPRESENTATION_TYPE == 'L' else BlockRepresentation()
        arch2.genotype = copy.deepcopy(arch1.genotype)
        arch2.idx = arch1.idx
        arch2.acc = arch1.acc
        arch2.loss = arch1.loss
        arch2.flops = arch1.flops
        arch2.num_params = arch1.num_params
        arch2.cpu_hours = arch1.cpu_hours
        arch2.trained_epochs = arch1.trained_epochs
        arch2.acc_hist = copy.copy(arch1.acc_hist)
        arch2.loss_hist = copy.copy(arch1.loss_hist)
        arch2.genoStr = arch1.genoStr
        arch2.gen_creation = arch1.gen_creation
        arch2.dP1 = arch1.dP1
        self.dP2 = arch1.dP2
        self.dBM = arch1.dBM
        arch1.P1Idx = arch2.P1Idx
        arch1.P2Idx = arch2.P2Idx
        arch1.P1IntegerEncoding = arch2.P1IntegerEncoding
        arch1.P2IntegerEncoding = arch2.P2IntegerEncoding
        arch1.before_mutationIdx = arch2.before_mutationIdx
        arch1.before_mutationIntegerEndcoding = arch2.before_mutationIntegerEndcoding
        arch1.P1acc = arch2.P1acc
        arch1.P2acc = arch2.P2acc
        arch1.BMacc = arch2.BMacc
        return arch2

    def flops_params_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                print(f'Processing file: {filename} for FLOPs and number of params calculation')
                filepath = os.path.join(folder_path, filename)
                df = pd.read_csv(filepath, encoding='utf-8')
                target_rows = df[(df['arch_status'] == 'BEST') & (df['FLOPs'].isna())]
                total = len(target_rows)
                arch_count = 0
                for idx in target_rows.index:
                    arch_count += 1
                    arch_ID = df.at[idx, 'ID']
                    print(Fore.LIGHTBLUE_EX + f'Processing arch {arch_ID}:  {arch_count}/{total} ============== ')
                    genotype = df.at[idx, 'Genotype']
                    flops, num_params = self.flops_params_gen_list_str(genotype)
                    df.at[idx, 'FLOPs'] = flops
                    df.at[idx, 'Num_Params'] = num_params
                    df.to_csv(filepath, index=False)
                    print(f'Saving progress in {filepath}')
                    
                os.rename(filepath, filepath.replace('.csv', '_flops_params.csv'))
                print(f'Completed processing {filename}. All {total} rows updated.\n')

    def determine_archs_flops_params(self, arch):
        if arch.idx not in self.flops_archs:
            print(Fore.CYAN + f'{arch.idx} FLOPS and NumParams calculation started')
            arch.flops = calculate_model_flops(arch.model)
            arch.num_params = calculate_model_params(arch.model)
            self.flops_archs[arch.idx] = [arch.flops, arch.num_params]
            print(Fore.CYAN + f'{arch.idx} FLOPS and NumParams calculation complete. {arch.flops}, {arch.num_params}')
        else:
            #If the architecture was already calculated, use the saved values
            print(Fore.LIGHTRED_EX + f'{arch.idx} FLOPS and NumParams already calculated. Using saved values')
            arch.flops = self.flops_archs[arch.idx][0]
            arch.num_params = self.flops_archs[arch.idx][1]
            print(Fore.LIGHTRED_EX + f'{arch.idx} FLOPS and NumParams: {arch.flops}, {arch.num_params}')
                            

    def get_lower_median_architecture_idx(self, listArchs):
        # Returns the lower median architecture based on accuracy.
        n = len(listArchs)
        lower_median_idx = (n - 1) // 2
        return lower_median_idx

    

    def genetic_operators(self):
        if self.search_strategy == 'GA':
            if self.crossover_type != 'NONE':
                print(Fore.GREEN + f'\nUsing Crossover {self.crossover_type}')
                self.generate_offspring() #CROSSOVER
                print(Fore.GREEN + 'Crossover complete\n')
            else: #NO CROSSOVER
                #Copy the parents to the children in case of only mutation. (Mutation only works with children and they are generated in the crossover function)
                print(Fore.RED + 'No Crossover operation selected. Copying parents to children')
                self.children = []
                for arch in self.pop:
                    #self.children.append(self.copy_arch(arch))
                    self.children.append(copy.copy(arch))
                for arch_child in self.children:
                    arch_child = self.make_child(arch_child, arch_child, arch_child) #Make a child of itself. This is to keep the same structure as the crossover function. Parents are itself
                print(Fore.RED + 'Copying parents to children complete\n')
            if self.mutation_type != 'NONE': 
                print(Fore.GREEN + '\nMutating offspring')
                self.mutate_children(self.mutation_type)
                print(Fore.GREEN + 'Mutating children done\n')
            else: #NO MUTATION
                print(Fore.RED + 'No Mutation operation selected. Children will not be mutated\n')
        else: #RANDOM SEARCH
            print(Fore.GREEN + f'\nGenerating random children {self.search_strategy}')
            self.children = []
            for i in range(self.NPOP):
                random_arch = self.random_individual()
                random_arch = self.train_model(random_arch)
                self.children.append(random_arch)
            print('Generating and training initial population complete')
            print(Fore.GREEN + 'Generating random children\n')
                        

    def child_better_parents(self, child):
        #Check if the child is better than both parents
        if child.acc > child.P1acc and child.acc > child.P2acc:
            return True
        else:
            return False

    def get_normalize_dataset(self):
        if ConfigClass.DATASET == 'CIFAR10':
            print(Fore.YELLOW + f'Loading CIFAR10 dataset' + Fore.RESET)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif ConfigClass.DATASET == 'CIFAR100':
            print(Fore.YELLOW + f'Loading CIFAR100 dataset' + Fore.RESET)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {ConfigClass.DATASET} dataset not recognized.' + Fore.RESET)
            x_train = None
        y_train = y_train.flatten()
        total_train_samples = x_train.shape[0]
        
        if ConfigClass.DATASET_PART == 1:
            subset_x_train = x_train
            subset_y_train = y_train
        else:
            subset_x_train,_,subset_y_train,_ = train_test_split(x_train, y_train, train_size = ConfigClass.DATASET_PART, stratify = y_train, random_state = 42)
        subset_y_train = subset_y_train.reshape(-1, 1)
        unique, counts = np.unique(subset_y_train, return_counts=True)
        print(dict(zip(unique, counts)))
        self.x_train = subset_x_train / 255.0
        self.x_test = x_test / 255.0
        self.y_train = subset_y_train.flatten()
        self.y_test = y_test.flatten()
        self.train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        self.train_datagen.fit(x_train)
        self.train_generator = self.train_datagen.flow(self.x_train, self.y_train, batch_size=Globals.BATCH_SIZE)

        self.validation_datagen = ImageDataGenerator()
        self.validation_generator = self.validation_datagen.flow(self.x_test, self.y_test, batch_size=Globals.BATCH_SIZE)


    def make_child(self, child, parent1, parent2):
        child.isChild = True
        child.isMutant = False
        child.P1Idx = parent1.idx
        child.P2Idx = parent2.idx
        child.P1IntegerEncoding = parent1.integer_encoding
        child.P2IntegerEncoding = parent2.integer_encoding
        child.before_mutationIdx = ''
        child.before_mutationIntegerEndcoding = None
        child.integer_encoding = child.genList_to_integer_vector()
        child.set_genoStr()
        child.dP1 = hamming_distance(child.integer_encoding, parent1.integer_encoding)
        child.dP2 = hamming_distance(child.integer_encoding, parent2.integer_encoding)
        child.dBM = -1

        child.P1acc = parent1.acc
        child.P2acc = parent2.acc
        return child

    def make_mutant(self, archM, archOriginal):
        archM.isChild = True
        archM.isMutant = True

        archM.P1Idx = archOriginal.P1Idx
        archM.P2Idx = archOriginal.P2Idx
        archM.P1IntegerEncoding = archOriginal.P1IntegerEncoding
        archM.P2IntegerEncoding = archOriginal.P2IntegerEncoding
        archM.before_mutationIdx = archOriginal.idx
        archM.before_mutationIntegerEndcoding = archOriginal.integer_encoding
        archM.integer_encoding = archM.genList_to_integer_vector()
        archM.dP1 = archOriginal.dP1
        archM.dP2 = archOriginal.dP2
        archM.BMacc = archOriginal.acc
        print(f'make_mutant {archM.idx = } {archM.integer_encoding} {archOriginal.idx = } {archOriginal.integer_encoding}')
        archM.dBM = hamming_distance(archM.integer_encoding, archOriginal.integer_encoding)

        archM.set_genoStr()
        return archM

    def make_parent(self, arch):
        #ind = copy.deepcopy(arch)
        arch.isChild = False
        arch.isMutant = False
        arch.P1Idx = ''
        arch.P2Idx = ''
        arch.P1IntegerEncoding = None
        arch.P2IntegerEncoding = None
        arch.before_mutationIdx = ''
        arch.before_mutationIntegerEndcoding = None
        arch.dP1 = -1
        arch.dP2 = -1
        arch.dBM = -1
        return arch

    def validate_architecture(self, gen_list, arch_idx = ''):
        #Determine if the architecture is valid before training. Check that there are no consecutive POOL layers
        #Returns a valid gen_list by changing the POOL to CONV
        if ConfigClass.REPRESENTATION_TYPE == 'B':
            return gen_list #No need to validate Block architectures
        print(f'Validating {arch_idx}:  {gen_list}')
        if len(gen_list) != LAYERS_CONSTANTS.SIZE_GENLIST:
            print(f'FATAL ERROR. ARCHITECTURE SIZE CHANGED: {len(gen_list)}, should be {LAYERS_CONSTANTS.SIZE_GENLIST}')
            return None
        return gen_list
    
    def change_block(self, child, mutation_type):
        mutator_obj = Mutator()
        arch = BlockRepresentation(genotypeObj = None)
        arch.genotype = copy.deepcopy(child.genotype)
        mutable_block_indexes = BLOCKS_CONSTANTS.MUTABLE_BCHANGEPARAM_INDEXES
        
        if mutation_type == 'MPAR':
            mutation_function = mutator_obj.mutate_block_parameters
        elif mutation_type == 'MSWAP':
            mutation_function = mutator_obj.mutate_block_swap
        else:
            print(f'ERROR: {mutation_type} Mutation type not recognized')
            mutation_function = None
            
        for block_idx in mutable_block_indexes:
            if random.random() < self.MUT_PROB:
                arch.genotype.gen_list = mutation_function(arch.genotype, block_idx)
                arch.set_genotype(arch.genotype)
        return arch

    def mutate_layer(self, child, mutation_type):
        #gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
        #Mutate each layer with probability MUT_PROB.
        #Save it in the genotype and update it.
        #Return the mutated architecture
        mutator_obj = Mutator()
        archMut = LayerRepresentation(genotypeObj = None)
        archMut.genotype = copy.deepcopy(child.genotype)
        if child.encoding == 'IV':
            if mutation_type == 'MPAR':
                layers_indexes = LAYERS_CONSTANTS.MUTABLE_LCHANGEPARAM_INDEXES #Only this indexes may change their parameters
                mutation_function = mutator_obj.mutate_layer_parameters
            elif mutation_type == 'MTYP':
                layers_indexes = LAYERS_CONSTANTS.MUTABLE_LCHANGETYPE_INDEXES #Only this indexes may change type
                mutation_function = mutator_obj.mutate_layer_type
            else:
                print(Fore.RED + f'ERROR: {mutation_type} Mutation type not recognized' + Fore.RESET)
                return None
            for layer_idx in layers_indexes:
                if random.random() < self.MUT_PROB:
                    archMut.genotype.gen_list = mutation_function(archMut.genotype, layer_idx)
                    archMut.set_genotype(archMut.genotype)
        elif child.encoding == 'BV':
            #The index of the last POOL is the maximum integer for the bitflip mutation
            MAX_INT = Globals.INDEXES_POOLS[-1]
            #Take the integer encoding and flip each bit with probability MUT_PROB
            if mutation_type == 'MBFLIP':
                #print(f'mutate layer 364 CHILD\n{child.genotype.gen_list}\n{child.integer_encoding}\n{child.binary_encoding}')
                archMut.binary_encoding = mutator_obj.mutate_bitflip(child.binary_encoding, MAX_INT)
                archMut.binary_encoding_to_integer_encoding()
                #print(f'\nmutate layer 367 MUTCHILD\n{archMut.genotype.gen_list}\n{archMut.integer_encoding}\n{archMut.binary_encoding}')
                archMut.genotype.gen_list = archMut.integer_vector_to_genList(archMut.integer_encoding)
                archMut.set_genotype(archMut.genotype)
            else:
                print(Fore.RED + f'ERROR: {mutation_type} Mutation type not recognized' + Fore.RESET)
                return None
        return archMut

   
    def mutate_children(self, mutation_type):
        if self.search_strategy == 'RANDOM':
            return
        
        for i in range(len(self.children)):
            name_before_mut = self.children[i].idx
            print(Fore.GREEN + f'MUTATING {self.children[i].idx} with {mutation_type} mutation')
            #print(self.children[i].integer_encoding)
            if ConfigClass.REPRESENTATION_TYPE == 'L':
                mutated_child = self.mutate_layer(self.children[i], mutation_type)
                
            elif ConfigClass.REPRESENTATION_TYPE == 'B':
                mutated_child = self.change_block(self.children[i], mutation_type)

            else:
                print(Fore.LIGHTRED_EX + f'ERROR (mutate_children): {self.representation_type} Representation type not recognized' + Fore.RESET)
                return None
            print(Fore.GREEN + f'MUTATING {self.children[i].idx} with {mutation_type} mutation complete\n')
            self.total_mut += 1
            mutated_child.idx = str(name_before_mut) + '[M]' #Add M to the index to indicate it was mutated      
            mutated_child = self.make_mutant(mutated_child, self.children[i])
            mutated_child = self.train_model(mutated_child)
            print('Assesing if the mutation was succesful')
            self.succ_mut += 1 if mutated_child.acc > self.children[i].acc else 0
            print('Assesing if the mutation was succesful completed')
            #self.children[i] = copy.deepcopy(mutated_child)
            self.children[i] = mutated_child
            #print(Fore.RED + f'MUTATING {self.children[i].idx} with {mutation_type} mutation complete')
            #print(self.children[i].integer_encoding)
            #print()
    
    def crossover(self, arch_obj1, arch_obj2):
        crossover_obj = Crossover()
        #Check what kind of crossover is being used
        if self.crossover_type == 'SPC':
            point = random.choice(TECNAS.REP_CONSTANTS.SPC_INDEXES)
            child1_arch, child2_arch = crossover_obj.single_point_crossover(arch_obj1, arch_obj2, point)
        elif self.crossover_type == 'TPC':
            [point1, point2] = random.sample(TECNAS.REP_CONSTANTS.SPC_INDEXES, 2)
            child1_arch, child2_arch = crossover_obj.two_point_crossover(arch_obj1, arch_obj2, point1, point2)
        elif self.crossover_type == 'UC':
            child1_arch, child2_arch = crossover_obj.uniform_crossover(arch_obj1, arch_obj2)
        else:
            print(f'UNRECOGNIZED CROSSOVER METHOD {self.crossover_type}')
            return None, None
        return child1_arch, child2_arch

    def random_parent_selection(self):
        [p1, p2] = random.sample(self.pop, 2)
        p1 = self.make_parent(p1)
        p2 = self.make_parent(p2)
        return (p1, p2)

    def generate_offspring(self):
        #reporter = ReportENAS()
        #Use Crossover or Random search
        self.children = []
        print(f'GA SEARCH STRATEGY')
        for _ in range(self.NPOP//2): #//2 because you add two children
            print('\nSelecting parents')
            parent1, parent2 = self.random_parent_selection()
            print(f'Parents: ' + Fore.GREEN + f'{parent1.idx}, {parent2.idx}')
            print(Fore.LIGHTBLUE_EX + f'Using crossover')
            child1, child2 = self.crossover(parent1, parent2)
            child1.genotype.gen_list = self.validate_architecture(child1.genotype.gen_list)
            child2.genotype.gen_list = self.validate_architecture(child2.genotype.gen_list)
            child1.set_genoStr()
            child2.set_genoStr()
            child1 = self.make_child(child1, parent1, parent2)
            child2 = self.make_child(child2, parent1, parent2)
            self.total_cross += 2
            print(f'Children: ' + Fore.LIGHTBLUE_EX + f'{child1.idx}, {child2.idx}')
            print('\nTraining children')
            child1 = self.train_model(child1)
            child2 = self.train_model(child2)
            print('Children training done')

            print('\nEvaluating if crossover was succesful')
            self.succ_cross += 1 if self.child_better_parents(child1) else 0
            self.succ_cross += 1 if self.child_better_parents(child2) else 0
            print('Evaluation done\n')
            
            self.children.append(child1)
            self.children.append(child2)
                    
    def create_model(self, arch_obj):
        layers_list = arch_obj.decode()
        if self.representation_type == 'L':
            model = models.Sequential(layers_list)
        elif self.representation_type == 'B':
            model = layers_list #For Block representation, the model is created in the decode function
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {self.representation_type} Representation type not recognized' + Fore.RESET)
            return None
        return model
    
    def random_genlist(self, num_dense = ConfigClass.DATASET_NUMDENSE_DICT):
        gen_list = []
        if self.representation_type == 'L':
            pool_count = 0
            max_pool_count = 3
            gen_list.append({'INP':Globals.INPUT_SIZE})
            gen_list.append(create_conv_layer())
            for _ in range(LAYERS_CONSTANTS.SIZE_GENLIST-LAYERS_CONSTANTS.NUM_FIXED_LAYERS):
                layer = random.choice(Globals.all_convs_pools)
                if get_key_from_value(layer, 'POOLMAX') or get_key_from_value(layer, 'POOLAVG'):
                    pool_count += 1
                    if pool_count > max_pool_count:
                        layer = create_conv_layer()
                gen_list.append(layer)
            gen_list.append({'FLATTEN':None})
            gen_list.append(create_dense_layer())
            gen_list.append(create_dense_layer(last_dense=True))
            gen_list = self.validate_architecture(gen_list)
        elif self.representation_type == 'B':
            gen_list.append({'INP':Globals.INPUT_SIZE})
            gen_list.append(create_conv_layer(nf = 64, ks = 3))
            random_blocks = random.choices(BlockRepresentation.all_blocks, k = ConfigBlocks.NBLOCKS)
            for block in random_blocks:
                gen_list.append(block)
            gen_list.append(create_globalAVG_layer())
            gen_list.append(create_dense_layer(last_dense = True))
            gen_list = BlockRepresentation.reset_add_pools(gen_list) #Add POOLMAX after 2 blocks

        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {self.representation_type} Representation type not recognized' + Fore.RESET)
            return None
        
        return gen_list

    def create_individual(self, idx = random.choice(ARCH_NAMES_LIST),  gen_list = None):
        #Create the inner layers, then add the input size at 0 and the last layers at the end
        genotype = Genotype(self.representation_type, 'IV', gen_list)

        if self.representation_type == 'L':
            arch_obj = LayerRepresentation('S', str(idx), genotypeObj=genotype)
        elif self.representation_type == 'B':
            arch_obj = BlockRepresentation('Sk', str(idx), genotypeObj=genotype)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR (TECNAS.create_individual): {self.representation_type} Representation type not recognized' + Fore.RESET)
            return None
        return arch_obj

    def random_individual(self):
        # TODO: Add more representations and encodings.
        #Create the inner layers, then add the input size at 0 and the last layers at the end
        
        gen_list = []
        gen_list = self.random_genlist()
        genotype = Genotype(self.representation_type, self.encoding_type, gen_list)
        idx = random.choice(ARCH_NAMES_LIST)

        if self.representation_type == 'L':
            arch_obj = LayerRepresentation(idx = str(idx), genotypeObj = genotype)
        elif self.representation_type == 'B':
            arch_obj = BlockRepresentation(idx = str(idx), genotypeObj = genotype)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {self.representation_type} Representation type not recognized' + Fore.RESET)
            return None
        return arch_obj

    def simulate_training(self, arch):
        print(f"Simulating trainining model {arch.idx}")
        # Simulate training and return the architecture with simulated accuracy and loss
        arch.acc = random.randint(0, 100)
        arch.loss = random.uniform(0, 1)
        arch.flops = random.randint(0, 100)
        arch.cpu_hours = random.uniform(0, 1)
        arch.num_params = random.randint(0, 100)
        arch.gen_creation = self.generation
        self.reporter.save_arch_info(self, arch)
        print(Fore.GREEN + f'Simulating Training {arch.idx} complete. Accuracy {arch.acc}\n')

    def training_with_surrogate(self, arch):
        print(f"Predicting model {arch.idx}")
        self.surrogate.load_arch(arch)
        arch.acc = self.surrogate.predict_arch(arch.idx, self.regressor_type)
        #arch_obj.model = self.create_model(arch_obj)
        arch.acc_hist = []
        arch.loss_hist = []
        arch.loss = np.nan     # Final validation loss
        arch.cpu_hours = np.nan            # Training time in CPU-hours
        arch.num_params = np.nan #calculate_model_params(model) # Total number of model parameters
        arch.flops = np.nan #calculate_model_flops(model)
        arch.trained_epochs = np.nan
        arch.gen_creation = self.generation
        self.reporter.save_arch_info(self, arch)
        print(Fore.GREEN + f'\nTraining {arch.idx} complete...\n')
    
    def real_training(self, arch, epochs = ConfigClass.EPOCHS, calculate_cm = False):
        start_time = time.time()
        with tf.device('/gpu:0'):
            tf.keras.backend.clear_session()
            arch.model = self.create_model(arch)
            #arch.model.summary()
            #arch_obj.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            arch.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=5e-4),  loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            #patience = 3 if epochs == 5 else 5
            #early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
            #steps_per_epoch = math.ceil(len(self.x_train) / BATCH_SIZE)
            #history = model.fit(self.train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=self.validation_generator,  callbacks=[early_stopping] )
            history = arch.model.fit(self.train_generator, batch_size=Globals.BATCH_SIZE, epochs=epochs, validation_data=self.validation_generator, callbacks=[reduce_lr], verbose = 1)
            #history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test, y_test), callbacks=[reduce_lr], verbose=1)
            # Evaluate the model on test data
            print(f'Evaluating {arch.idx} on test data...')
            test_loss, test_acc = arch.model.evaluate(self.x_test, self.y_test, verbose=0)
            print(f'\nTraining {arch.idx} complete')
            

        # Calculate and print CPU-Hours
        
        arch.acc_hist = history.history['val_accuracy']
        arch.loss_hist = history.history['val_loss']
        arch.acc = history.history['val_accuracy'][-1]  # Final validation accuracy
        arch.loss = history.history['val_loss'][-1]     # Final validation loss
        arch.cpu_hours = training_time_hours            # Training time in CPU-hours
        #arch_obj.num_params = calculate_model_params(model) # Total number of model parameters
        #arch_obj.flops = calculate_model_flops(model)
        arch.trained_epochs = len(history.history['loss'])
        arch.gen_creation = self.generation

        if calculate_cm == True:
            print(Fore.YELLOW + f'Calculating confusion matrix for {arch.idx}')
            self.compute_confusion_matrix(arch)
            self.compute_metrics_from_cm(arch)
            print(Fore.YELLOW + f'Confusion matrix for {arch.idx} calculated')
            print(Fore.YELLOW + f'Calculating FLOPs and number of parameters for {arch.idx}')
            flops, num_params = self.flops_params_gen_list_str(str(arch.genotype.gen_list))
            arch.flops = flops
            arch.num_params = num_params
            print(Fore.YELLOW + f'FLOPs and number of parameters for {arch.idx} calculated: {arch.flops}, {arch.num_params}')
            

        self.reporter.save_arch_info(self, arch)
        print(Fore.GREEN + f'\nTraining {arch.idx} complete...\n')
        end_time = time.time()
        elapsed_min = (end_time - start_time)/60
        print(Fore.LIGHTBLUE_EX + f"{elapsed_min:.1f} min.")
        return arch
    
    
    
    
    def train_model(self, arch, calculate_cm = False, epochs = ConfigClass.EPOCHS, print_status = True):
        ast_bar = self.ast_bar
        if print_status == True:
            print(Fore.LIGHTBLUE_EX + f'\n{ast_bar} EXECUTION {self.exec}/{ConfigClass.EXECUTIONS} GENERATION {self.generation}/{ConfigClass.GENERATIONS} {self.crossover_type} {self.mutation_type} {self.arch_count+1}/{ConfigClass.TOTAL_ARCH} architectures {ast_bar}')
        print(Fore.YELLOW + f'{ConfigClass.DATASET} {ConfigClass.DATASET_PART*100}% |Repr Encoding: {self.representation_type}-{self.encoding_type}| |Surrogate: {self.SURROGATE}| |Train: {self.TRAIN}| |Simulate: {self.SIMULATE}| {self.experiment_folder}' + Fore.RESET)
        
        self.arch_count += 1
        memory = psutil.virtual_memory()
        # Print the percentage of RAM used
        print(Fore.RED + f"RAM used: {memory.percent}%\n")
        print(Fore.GREEN + f'\nTraining {arch.idx}...')
        if ConfigClass.REPRESENTATION_TYPE == 'B': #
            arch.genotype.gen_list = BlockRepresentation.reset_add_pools(arch.genotype.gen_list)
        print(arch.genotype.gen_list)
        print(arch.integer_encoding)
        print(arch.binary_encoding)
        print()
        if self.SIMULATE == True:
            self.simulate_training(arch)
        elif self.SURROGATE == True:
            self.training_with_surrogate(arch)
        elif self.TRAIN == True:
            self.real_training(arch, epochs = epochs, calculate_cm = calculate_cm)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: No training method selected' + Fore.RESET)
            return None
        return arch
    
    def set_arch_statuses(self):
        #Identify BEST, WORST AND MEDIAN. Save their info and reset their status. Also, calculate penalization for BEST.
        
        #Saving info about the WORST and MEDIAN ARCHITECTURES (BEFORE ELITISM) -------------------------- CHECKING THIS
        #Also store the worst arch and median
        self.sorted_pool[-1].archStatus = 'WORST'
        lower_median_idx = self.get_lower_median_architecture_idx(self.sorted_pool)
        self.sorted_pool[lower_median_idx].archStatus = 'MEDIAN'
        self.sorted_pool[0].archStatus = 'BEST'
        self.reporter.save_arch_info(self, self.sorted_pool[-1], resetInfo = True)
        self.reporter.save_arch_info(self, self.sorted_pool[lower_median_idx], resetInfo = True)
        self.sorted_pool[-1].archStatus = ''
        self.sorted_pool[lower_median_idx].archStatus = ''
        self.check_penalization(self.sorted_pool[0])
        #resetInfo is used for resetting some info about the best arch. saveRatios is for saving succ cross and succ mut.
        self.reporter.save_arch_info(self, self.sorted_pool[0], resetInfo = True) 
        self.sorted_pool[0].archStatus = ''
                    
    def select_elitism(self):
        #Based on the SORT_ARCHS variable, select the best architectures from the sorted pool.

        if ConfigClass.SORT_ARCHS == 'SUPERIOR':
            begin = 0
            end = self.NPOP
        elif ConfigClass.SORT_ARCHS == 'INFERIOR':
            begin = len(self.sorted_pool) - self.NPOP
            end = len(self.sorted_pool)
        elif ConfigClass.SORT_ARCHS == 'MIDDLE':
            lower_median_idx = self.get_lower_median_architecture_idx(self.sorted_pool)
            begin = lower_median_idx - self.NPOP//2
            end = lower_median_idx + self.NPOP//2
        else:
            print(f'ERROR: {ConfigClass.SORT_ARCHS} is not a valid sorting method')

        self.pop = self.sorted_pool[begin:end]

    def check_penalization(self, arch_obj):
        if arch_obj.gen_creation == self.generation:
            self.penalization = 0
        else:
            self.penalization = 1

    def initialize_pop(self):
            self.pop = []
            self.pool = []
            self.children = []
            print('Generating and training initial population')
            for i in range(self.NPOP):
                random_arch = self.random_individual()
                random_arch = self.train_model(random_arch)
                self.pop.append(random_arch)
            print('Generating and training initial population complete')

    def ENAS(self):
        for crossover_type in ConfigClass.crossover_types_list:
            self.crossover_type = crossover_type
            for mutation_type in ConfigClass.mutation_types_list:
                self.mutation_type = mutation_type
                if crossover_type == 'NONE' and mutation_type == 'NONE':
                    self.search_strategy = 'RANDOM'
                else:
                    self.search_strategy = 'GA'
                if ConfigClass.RANDOMIZE_SEED == True:
                    self.local_seed = random.randint(7,23357) #SEED
                else:
                    self.local_seed = ConfigClass.SEED
                random.seed(self.local_seed)
                np.random.seed(self.local_seed)
                #tf.random.set_seed(self.local_seed)

                for e in range(1,ConfigClass.EXECUTIONS+1):
                    self.succ_cross = 0
                    self.succ_mut = 0
                    self.total_cross = 0
                    self.total_mut = 0
                    self.exec = e
                    self.trained_archs = {}
                    print('\nInitializing population')
                    self.generation = 1
                    self.initialize_pop()
                    print('Initialization Done\n')
                    for g in range(1,ConfigClass.GENERATIONS+1):
                        self.pool = []
                        self.generation = g
                        print(Fore.LIGHTYELLOW_EX + 'Generating offspring')
                        self.genetic_operators()
                        print(Fore.LIGHTYELLOW_EX + 'Generating offspring complete\n')
                        self.pool = self.pop + self.children
                        self.sorted_pool = sorted(self.pool, key=lambda obj: obj.acc, reverse = True) #Best archs first
                        print('\nIdentifying BEST, MEDIAN and WORST architectures')
                        self.set_arch_statuses()
                        print('Identifying BEST, MEDIAN and WORST architectures complete\n')
                        print('\nSelecting best parents and children for the new population')
                        self.select_elitism() #Here, WORST AND MEDIAN ARCHS ARE IDENTIFIED
                        print('Selecting best parents and children for the new population complete\n')
                        print('\nBest architecture is: ')
                        print(self.pop[0].idx, self.pop[0].acc, self.pop[0].archStatus)
                        #Calculate flops and num_params for the best architecture
                        #if SIMULATE == False:
                        #    self.determine_archs_flops_params(self.sorted_pop[0])                                                      
                        
                    self.local_seed += 1
                    print('\nSaving histories')
                    self.accuracy_histories.append(self.pop[0].acc_hist)
                    self.loss_histories.append(self.pop[0].loss_hist)

                    self.best_acc_list.append(self.pop[0].acc)
                    self.loss_list.append(self.pop[0].loss)
                    self.cpu_hours_list.append(self.pop[0].cpu_hours)
                    self.num_params_list.append(self.pop[0].num_params)
                    self.flops_list.append(self.pop[0].flops)
                    self.best_archs.append(self.pop[0])
                    print('Done========================')
        print('Ranking architectures')
        rank_archs(self.reporter.path_report)
        print('Ranking finished\n')


    def __init__(self, representation_type = ConfigClass.REPRESENTATION_TYPE, encoding_type = ConfigClass.ENCODING_TYPE, experiment_folder = '', regressor_type = -1):
        self.ast_bar = 50*'+'
        self.representation_type = representation_type #Layer (L), Block (B)
        self.encoding_type = encoding_type
        self.NPOP = ConfigClass.MAIN_NPOP
        self.MUT_PROB = ConfigClass.MUT_PROB
        self.crossover_types_list = ConfigClass.crossover_types_list
        self.crossover_type = None
        self.mutation_type = None
        self.search_strategy = None
        self.reporter = ReportENAS()
        self.generation = 0
        self.penalization = 0 #0 if the best architecture was found in the generation it was created, otherwise 1.
        self.exect = 0
        self.arch_count = 0
        self.succ_mut = 0
        self.succ_cross = 0
        self.total_mut = 0
        self.total_cross = 0
        self.pop = []
        self.children = []
        self.pool = [] #Parents + children
        self.trained_parents = []
        self.flops_archs = {} #Dictionary with architectures whose FLOPs and NumParams were already calculated
        self.accuracy_histories = [] #To plot all accuracies through epochs
        self.loss_histories = [] #To plot all accuracies through epochs
        self.NUM_DENSE = ConfigClass.DATASET_NUMDENSE_DICT[ConfigClass.DATASET]
        

        self.report_columns = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs",
                   "Acc_mean", "Loss_mean", "CPU_hrs_mean", "Num_params_mean",
                   "FLOPs_mean", "Acc_std", "Loss_std", "CPU_hrs_std", "Num_params_std",
                   "FLOPs_std"]
        self.bar_columns = ['Acc_mean', "Loss_mean", "CPU_hrs_mean", "Num_params_mean", "FLOPs_mean"]
        self.box_columns = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs"]
        self.bar_median_columns = ['Acc_mean', "Loss_mean"]
        self.box_median_columns = ["Best_accuracy", "Loss"]
        #Lists for report
        self.best_accHist_list = []
        self.best_acc_list = []
        self.loss_list = []
        self.cpu_hours_list = []
        self.num_params_list = []
        self.flops_list = []

        self.best_archs = [] #For finding the median
        self.accuracy_median_histories = [] #To plot all accuracies through epochs, median arch
        self.loss_median_histories =[]

        self.general_report_filenames = [f'GA_L_CHANGE_TYPE', f'GA_L_MODIFY_PARAMS', f'RANDOM_NONE']
        self.medians_report_filenames = [f'GA_L_CHANGE_TYPE_MEDIAN', f'GA_L_MODIFY_PARAMS_MEDIAN', f'RANDOM_NONE_MEDIAN']
        self.filename = f'{self.search_strategy}_mutation_type' #For report and plot filenames

        self.experiment_folder = experiment_folder
        self.regressor_type = regressor_type
        self.regressor_folder = os.path.join(path, 'results', experiment_folder, 'regressors')
        ensure_folder_exists(self.regressor_folder)
        self.regressor_type = -1 if ConfigClass.SURROGATE == False else regressor_type
        if self.regressor_type >= 0:
            self.surrogate = Surrogate_ENAS(self.regressor_type, self.regressor_folder)

        self.SURROGATE = ConfigClass.SURROGATE
        self.TRAIN = ConfigClass.TRAIN
        self.SIMULATE = ConfigClass.SIMULATE
        self.REPORT_ARCH = ConfigClass.REPORT_ARCH

        if self.TRAIN == True:
            self.get_normalize_dataset()
        
    
'''
gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[128,'relu']}, {'LAST_DENSE':[10,'softmax']}]
archL = LayerRepresentation(genotypeObj = Genotype(rep_type='L', encoding_type='IV', gen_list=gen_list))
#print(archL)

print(archL.integer_encoding)
archL.integer_encoding_to_binary_encoding(10) #Usar N.bit_length()
print(archL.binary_encoding)

mut = Mutator()
#print(archL.binary_encoding[1])
archL.binary_encoding[1] = mut.mutate_bitflip(archL.binary_encoding[1], 1)
print(archL.binary_encoding)
archL.binary_encoding_to_integer_encoding()
print(archL.integer_encoding)

[0, 8, 2, 8, 10]
['0000', '1100', '0011', '1100', '1111']
'00001100001111001111'
'00001000001111001111'
['0000', '1000', '0011', '1100', '1111']
[0, 15, 2, 8, 10]
'''
'''
0: {'CONV': [32, 3]}
1: {'CONV': [32, 5]}
2: {'CONV': [64, 3]}
3: {'CONV': [64, 5]}
4: {'CONV': [128, 3]}
5: {'CONV': [128, 5]}
6: {'CONV': [256, 3]}
7: {'CONV': [256, 5]}
8: {'POOLMAX': [-1, 2]}
9: {'POOLMAX': [-1, 3]}
10: {'DENSE': [128, 'relu']}
11: {'DENSE': [256, 'relu']}
12: {'DENSE': [512, 'relu']}
'''



'''
0: ({'CONV': [32, 3]}, {'CONV': [32, 3]})
1: ({'CONV': [32, 3]}, {'CONV': [32, 5]})
2: ({'CONV': [32, 3]}, {'CONV': [64, 3]})
3: ({'CONV': [32, 3]}, {'CONV': [64, 5]})
4: ({'CONV': [32, 3]}, {'CONV': [128, 3]})
5: ({'CONV': [32, 3]}, {'CONV': [128, 5]})
6: ({'CONV': [32, 3]}, {'CONV': [256, 3]})
7: ({'CONV': [32, 3]}, {'CONV': [256, 5]})
8: ({'CONV': [32, 5]}, {'CONV': [32, 3]})
9: ({'CONV': [32, 5]}, {'CONV': [32, 5]})
10: ({'CONV': [32, 5]}, {'CONV': [64, 3]})
11: ({'CONV': [32, 5]}, {'CONV': [64, 5]})
12: ({'CONV': [32, 5]}, {'CONV': [128, 3]})
13: ({'CONV': [32, 5]}, {'CONV': [128, 5]})
14: ({'CONV': [32, 5]}, {'CONV': [256, 3]})
15: ({'CONV': [32, 5]}, {'CONV': [256, 5]})
16: ({'CONV': [64, 3]}, {'CONV': [32, 3]})
17: ({'CONV': [64, 3]}, {'CONV': [32, 5]})
18: ({'CONV': [64, 3]}, {'CONV': [64, 3]})
19: ({'CONV': [64, 3]}, {'CONV': [64, 5]})
20: ({'CONV': [64, 3]}, {'CONV': [128, 3]})
21: ({'CONV': [64, 3]}, {'CONV': [128, 5]})
22: ({'CONV': [64, 3]}, {'CONV': [256, 3]})
23: ({'CONV': [64, 3]}, {'CONV': [256, 5]})
24: ({'CONV': [64, 5]}, {'CONV': [32, 3]})
25: ({'CONV': [64, 5]}, {'CONV': [32, 5]})
26: ({'CONV': [64, 5]}, {'CONV': [64, 3]})
27: ({'CONV': [64, 5]}, {'CONV': [64, 5]})
28: ({'CONV': [64, 5]}, {'CONV': [128, 3]})
29: ({'CONV': [64, 5]}, {'CONV': [128, 5]})
30: ({'CONV': [64, 5]}, {'CONV': [256, 3]})
31: ({'CONV': [64, 5]}, {'CONV': [256, 5]})
32: ({'CONV': [128, 3]}, {'CONV': [32, 3]})
33: ({'CONV': [128, 3]}, {'CONV': [32, 5]})
34: ({'CONV': [128, 3]}, {'CONV': [64, 3]})
35: ({'CONV': [128, 3]}, {'CONV': [64, 5]})
36: ({'CONV': [128, 3]}, {'CONV': [128, 3]})
37: ({'CONV': [128, 3]}, {'CONV': [128, 5]})
38: ({'CONV': [128, 3]}, {'CONV': [256, 3]})
39: ({'CONV': [128, 3]}, {'CONV': [256, 5]})
40: ({'CONV': [128, 5]}, {'CONV': [32, 3]})
41: ({'CONV': [128, 5]}, {'CONV': [32, 5]})
42: ({'CONV': [128, 5]}, {'CONV': [64, 3]})
43: ({'CONV': [128, 5]}, {'CONV': [64, 5]})
44: ({'CONV': [128, 5]}, {'CONV': [128, 3]})
45: ({'CONV': [128, 5]}, {'CONV': [128, 5]})
46: ({'CONV': [128, 5]}, {'CONV': [256, 3]})
47: ({'CONV': [128, 5]}, {'CONV': [256, 5]})
48: ({'CONV': [256, 3]}, {'CONV': [32, 3]})
49: ({'CONV': [256, 3]}, {'CONV': [32, 5]})
50: ({'CONV': [256, 3]}, {'CONV': [64, 3]})
51: ({'CONV': [256, 3]}, {'CONV': [64, 5]})
52: ({'CONV': [256, 3]}, {'CONV': [128, 3]})
53: ({'CONV': [256, 3]}, {'CONV': [128, 5]})
54: ({'CONV': [256, 3]}, {'CONV': [256, 3]})
55: ({'CONV': [256, 3]}, {'CONV': [256, 5]})
56: ({'CONV': [256, 5]}, {'CONV': [32, 3]})
57: ({'CONV': [256, 5]}, {'CONV': [32, 5]})
58: ({'CONV': [256, 5]}, {'CONV': [64, 3]})
59: ({'CONV': [256, 5]}, {'CONV': [64, 5]})
60: ({'CONV': [256, 5]}, {'CONV': [128, 3]})
61: ({'CONV': [256, 5]}, {'CONV': [128, 5]})
62: ({'CONV': [256, 5]}, {'CONV': [256, 3]})
63: ({'CONV': [256, 5]}, {'CONV': [256, 5]})
'''