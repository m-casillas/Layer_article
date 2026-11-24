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
import numpy

from TECNAS_AOS import TECNAS_AOS
from TECNAS_NSGAII import TECNAS_NSGAII
from Status import Status
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
from itertools import product
import time
import joblib
import csv


"""# TECNAS Classs"""

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#genotype = Genotype('L', 'INT', gen_list)
#arch = Architecture('X', 0, 0, 0, 0, 1, genotype)

#TECNAS is a class that performs evoltuionary neural architecture search, using genetic operators
class TECNAS:
    if ConfigClass.REPRESENTATION_TYPE == 'L':
        REP_CONSTANTS = LAYERS_CONSTANTS
    elif ConfigClass.REPRESENTATION_TYPE == 'B':
        REP_CONSTANTS = BLOCKS_CONSTANTS
        REP_CONSTANTS.SPC_INDEXES = list(range(REP_CONSTANTS.SIZE_BLOCKGEN*2)) if config_tecnas.INTEGER_ENCODING2 else REP_CONSTANTS.SPC_INDEXES

    def validate_encoding_genetic_operators(self):
        valid_cross = {'BIN': ['SPC', 'TPC', 'UC', 'NONE'], 'INT': ['SPC', 'TPC', 'UC', 'NONE'], 'REAL': ['SPC', 'TPC', 'UC', 'SBX', 'NONE']}
        valid_mut = {'BIN': ['MBFLIP', 'NONE'], 'INT': ['MSWAP', 'MPAR', 'NONE'], 'REAL': ['MPOLY', 'MNUF', 'NONE']}
        # CHeck from the dictionary if the current encoding type has valid crossover and mutation types
        if self.encoding_type not in valid_cross or self.encoding_type not in valid_mut:
            print(f"{Fore.RED} Error: Invalid encoding type {self.encoding_type}. Valid types are: {list(valid_cross.keys())}{Fore.RESET}")
            return False
        if self.crossover_type not in valid_cross[self.encoding_type]:
            print(f"{Fore.RED} Error: Invalid crossover type {self.crossover_type} for encoding type {self.encoding_type}. Valid types are: {valid_cross[self.encoding_type]}{Fore.RESET}")
            return False
        if self.mutation_type not in valid_mut[self.encoding_type]:
            print(f"{Fore.RED} Error: Invalid mutation type {self.mutation_type} for encoding type {self.encoding_type}. Valid types are: {valid_mut[self.encoding_type]}{Fore.RESET}")
            return False
        return True
        

    def vector_to_arch(self, intvec = None, arch_obj1_idx = None, arch_obj2_idx = None, is_first_child = False, is_mutant = False):
        #intvec is integer_encoding only.
        idx1 = arch_obj1_idx+'|'+arch_obj2_idx+'(1)' if is_first_child else arch_obj1_idx+'|'+arch_obj2_idx+'(2)'
        idx1 = Architecture.set_arch_name(idx1)
        gen_list = BlockRepresentation.integer_vector_to_genList(intvec)
        arch = BlockRepresentation(idx = idx1, genotypeObj = Genotype('B', ConfigClass.ENCODING_TYPE, gen_list), INTEGER_ENCODING2 = self.INTEGER_ENCODING2)
        arch.integer_encoding, arch.integer_encoding2, arch.integer_encoding2_flattened, arch.real_encoding, arch.real_encoding2, arch.real_encoding2_flattened, arch.binary_encoding, arch.binary_encoding2_flattened = BlockRepresentation.update_vector_encodings(int_vec = intvec, INTEGER_ENCODING2 = self.INTEGER_ENCODING2, binary_encoding = True if self.encoding_type == 'BIN' else False)                            
        return arch

    
    def SCN_stats_histogram(self):
        cross_mut_str = f'{self.crossover_type}_{self.mutation_type}'
        cross_mut_str = self.REPR_ENC_STR + '_' + self.search_name
        def save_histogram(data=[], scn_type='', stats=None):
            plt.figure()
            plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(stats['min'], color='black', linestyle='-', linewidth=1,label=f'Min = {stats["min"]:.4f}')
            plt.axvline(stats['max'], color='black', linestyle='-', linewidth=1,label=f'Max = {stats["max"]:.4f}')
            plt.axvline(stats['mean'], color='blue', linestyle='-', linewidth=1, label=f'Mean = {stats["mean"]:.4f}')
            plt.axvline(stats['median'], color='red', linestyle='-', linewidth=1, label=f'Median = {stats["std"]:.4f}')
            for p, value in stats['percentiles'].items():
                color = 'green' if p < 50 else 'orange'
                linestyle = '--'
                linewidth = 1
                label = f'P{p} = {value:.4f}'
                plt.axvline(value, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            plt.title(f'{cross_mut_str} SCN_{scn_type} Histogram\nExecution {self.exec}')
            plt.xlabel(f'SCN_{scn_type} Value')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.legend(loc='best')
            hist_folder = os.path.join(self.pop[0].path_folder, 'SCN_stats')
            os.makedirs(hist_folder, exist_ok=True)
            hist_path = os.path.join(hist_folder, f'{cross_mut_str}_SCN{scn_type}_histogram_E{self.exec}.png')
            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
            plt.close()

        def get_stats(data=[], scn_type='', percentiles=[5, 10, 25, 50, 75, 90, 95, 99], bins=30):
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            median = np.median(data)
            percentile_values = np.percentile(data, percentiles)
            frequencies, bin_edges = np.histogram(data, bins=bins)
            frequency_table = []
            for i in range(len(frequencies)):
                frequency_table.append({'bin_min': bin_edges[i],'bin_max': bin_edges[i + 1], 'frequency': frequencies[i]})
            path_folder = self.pop[0].path_folder
            path_folder_file = os.path.join(path_folder, 'SCN_stats', f'{cross_mut_str}_SCN{scn_type}_stats.csv')
            os.makedirs(os.path.dirname(path_folder_file), exist_ok=True)
            file_exists = os.path.isfile(path_folder_file)
            with open(path_folder_file, 'a') as f:
                if not file_exists:
                    header = 'execution,search_name,distance_type,SCN,INTEGER_ENCODING2,mean,std,min,max,median'
                    header += ''.join([f',perc_{p}' for p in percentiles])
                    f.write(header + '\n')
                row = f'{self.exec},{self.search_name},{self.distance_type},SCN{scn_type},{self.INTEGER_ENCODING2},{mean},{std},{min_val},{max_val},{median}'
                row += ''.join([f',{value}' for value in percentile_values])
                f.write(row + '\n')
            frequency_path = os.path.join(path_folder, 'SCN_stats', f'{cross_mut_str}_SCN{scn_type}_frequencies.csv')
            frequency_file_exists = os.path.isfile(frequency_path)
            with open(frequency_path, 'a') as f:
                if not frequency_file_exists:
                    f.write('execution,search_name,distance_type,SCN,INTEGER_ENCODING2,bin_min,bin_max,frequency\n')
                for item in frequency_table:
                    f.write(f'{self.exec},{self.search_name},{self.distance_type},SCN{scn_type},{self.INTEGER_ENCODING2},{item["bin_min"]},{item["bin_max"]},{item["frequency"]}\n')
            return {'mean': mean, 'std': std,'min': min_val,'max': max_val,'median': median, 'percentiles': dict(zip(percentiles, percentile_values)), 'frequency_table': frequency_table}

        def get_stats2(data=[], scn_type='', percentiles=[5, 10, 25, 75, 90, 99]):
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            median = np.median(data)
            # Calculate requested percentiles
            percentile_values = np.percentile(data, percentiles)
            path_folder = self.pop[0].path_folder
            path_folder_file = os.path.join(path_folder, 'SCN_stats', f'{cross_mut_str}_SCN{scn_type}_stats.csv')
            os.makedirs(os.path.dirname(path_folder_file), exist_ok=True)
            file_exists = os.path.isfile(path_folder_file)
            with open(path_folder_file, 'a') as f:
                # Write header only if file is new
                if not file_exists:
                    header = ('execution,search_name,distance_type,SCN,INTEGER_ENCODING2,'
                              'mean,std,min,max,median' )
                    header += ''.join([f',perc_{p}' for p in percentiles])
                    f.write(header + '\n')
                row = (
                    f'{self.exec},{self.search_name},{self.distance_type},'
                    f'SCN{scn_type},{self.INTEGER_ENCODING2},'
                    f'{mean},{std},{min_val},{max_val},{median}'
                )
                row += ''.join([f',{value}' for value in percentile_values])
                f.write(row + '\n')
            return {'mean': mean, 'std': std, 'min': min_val, 'max': max_val, 'median': median, 'percentiles': dict(zip(percentiles, percentile_values))}
        percentiles = [5, 10, 25, 75, 90, 95, 99]
        stats_scn = get_stats(data = self.SCNG_history, scn_type = 'G', percentiles=percentiles) if self.calculate_SCNG else None
        save_histogram(data = self.SCNG_history, scn_type = 'G', stats=stats_scn) if self.calculate_SCNG else None
        stats_scn = get_stats(data = self.SCNF_history, scn_type = 'F', percentiles=percentiles) if self.calculate_SCNF else None
        save_histogram(data = self.SCNF_history, scn_type = 'F', stats=stats_scn) if self.calculate_SCNF else None
        
        

    
    def lognormalize_objectives(self, objective):
        accuracy, flops, params = objective
        min_acc = 0.0
        max_acc = 0.75 if self.representation_type == 'B' else 0.94
        min_flops = self.MINFLOPS
        max_flops = self.MAXFLOPS
        min_params = self.MINPARAMS
        max_params = self.MAXPARAMS
        # Accuracy: linear normalization
        acc_range = max_acc - min_acc if max_acc != min_acc else 1.0
        normalized_acc = np.clip( (accuracy - min_acc) / acc_range, 0.0, 1.0)
        # FLOPs: log normalization
        if min_flops > 0 and max_flops > min_flops:
            log_min_flops = np.log(min_flops)
            log_max_flops = np.log(max_flops)
            normalized_flops = np.clip((np.log(flops) - log_min_flops) / (log_max_flops - log_min_flops), 0.0, 1.0 )
        else:
            normalized_flops = 0.0
        # Parameters: log normalization
        if min_params > 0 and max_params > min_params:
            log_min_params = np.log(min_params)
            log_max_params = np.log(max_params)
            normalized_params = np.clip((np.log(params) - log_min_params) / (log_max_params - log_min_params),0.0,1.0)
        else:
            normalized_params = 0.0
        return (normalized_acc, normalized_flops, normalized_params)

    def normalize_objectives(self, objective):
        accuracy, flops, params = objective
        min_acc = 0.0
        max_acc = 0.75 if self.representation_type == 'B' else 0.94
        min_flops = self.MINFLOPS
        max_flops = self.MAXFLOPS
        min_params = self.MINPARAMS
        max_params = self.MAXPARAMS
        acc_range = max_acc - min_acc if max_acc != min_acc else 1.0
        flops_range = max_flops - min_flops if max_flops != min_flops else 1.0
        params_range = max_params - min_params if max_params != min_params else 1.0
        normalized_acc = np.clip( (accuracy - min_acc) / acc_range, 0.0, 1.0 )
        normalized_flops = np.clip((flops - min_flops) / flops_range, 0.0,1.0 )
        normalized_params = np.clip((params - min_params) / params_range, 0.0, 1.0)
        return (normalized_acc,normalized_flops,normalized_params)

    def save_encoding_vectors_history(self, architecture_set):
            for arch in architecture_set:
                vec = self.get_vector_for_distances(arch = arch)
                self.encoding_vectors_history.add(tuple(vec))

    def save_encoding_vectors_history2(self, architecture_set):
        for arch in architecture_set:
            if self.encoding_type == 'BIN':
                self.encoding_vectors_history.add(tuple(''.join(arch.binary_encoding2_flattened) if self.INTEGER_ENCODING2 else ''.join(arch.binary_encoding)))
            else:
                self.encoding_vectors_history.add(tuple(arch.integer_encoding2_flattened if self.INTEGER_ENCODING2 else arch.integer_encoding))
        print(f'Save encoding vectors history')
        #print(self.encoding_vectors_history)

    def save_objectives_history(self, architecture_set):
        for arch in architecture_set:
            self.objectives_history.add(self.lognormalize_objectives((arch.acc, arch.flops, arch.num_params)))

    def calculate_SCN(self, SCN_G = False, SCN_F = False):
            s = time.time()
            N = len(self.pop)
            color = Fore.GREEN 
            scnG = None
            scnF = None
            labelG = "G" if SCN_G else ""
            labelF = "F" if SCN_F else ""
            self.NEXPLOIT_SCNG = 0
            self.NEXPLORE_SCNG = 0
            self.NEXPLOIT_SCNF = 0
            self.NEXPLORE_SCNF = 0
            print1(f'{color} Calculating SCN_({labelG})({labelF}) for {len(self.pop)} architectures{Fore.RESET} ')
            for iarch, arch in enumerate(self.pop):
                if iarch%10 == 0:
                    print1(f'{color} Calculating SCN_({labelG})({labelF}) for architecture {iarch}/{len(self.pop)}{Fore.RESET}')
                    print1(f'{color} #GIndividuals:{len(self.encoding_vectors_history)} #FIndividuals:{len(self.objectives_history)}{Fore.RESET}')
                if SCN_G:
                    print2(f'Calculate SCNG')
                    print2(f'{arch.integer_encoding = }')
                    print2(f'{arch.integer_encoding2_flattened = }')
                    print2(f'{arch.binary_encoding = }')
                    print2(f'{arch.binary_encoding2_flattened = }')
                    print2(f'{arch.real_encoding = }')
                    print2(f'{arch.real_encoding2_flattened = }')
                    vec = self.get_vector_for_distances(arch = arch)
                    num_type = np.int16
                    num_type = np.float64 if (self.encoding_type == 'REAL') and (self.INTEGER_ENCODING2) else num_type
                    vec = np.asarray(vec, dtype=num_type)
                    history = np.asarray(list(self.encoding_vectors_history), dtype=num_type)    
                    scnG = self.calculate_SCN_G(vec = vec, history = history, distance_type = self.distance_type)
                    self.SCNG_history.append(scnG)
                    if scnG <= self.SCNG_THRESHOLD: 
                        self.NEXPLOIT_SCNG += 1
                    else:
                        self.NEXPLORE_SCNG += 1
                    self.perc_explore_SCNG = self.NEXPLORE_SCNG/N
                    self.perc_exploit_SCNG = self.NEXPLOIT_SCNG/N
                if SCN_F:
                    scnF = self.calculate_SCN_F((arch.acc, arch.flops, arch.num_params))
                    self.SCNF_history.append(scnF)
                    if scnF <= self.SCNF_THRESHOLD:
                        self.NEXPLOIT_SCNF += 1
                    else:
                        self.NEXPLORE_SCNF += 1
                    self.perc_explore_SCNF = self.NEXPLORE_SCNF/N
                    self.perc_exploit_SCNF = self.NEXPLOIT_SCNF/N
            print1(f'{color} Calculating SCN_({labelG})({labelF}) complete{Fore.RESET}')
            f = time.time()
            print1(f'\n{color} SCN_({labelG})({labelF}) calculation time: {f-s:.2f} seconds{Fore.RESET}')


    def calculate_SCN_F2(self, arch_objective = None):
        #Vectorized form using numpy
        #Calculate Similariy of Closest Neighbbor of current individual with all individuals through all generations.
        np_objectives_history = np.array(list(self.objectives_history))
        np_normalized_arch_objective = np.array(self.lognormalize_objectives(arch_objective))
        #Delete arch_objective from history if it exists to avoid self-comparison
        np_objectives_history = np_objectives_history[~np.all(np_objectives_history == np_normalized_arch_objective, axis=1)]
        dist_vec = np.linalg.norm(np_objectives_history - np_normalized_arch_objective, axis=1)
        return np.min(dist_vec)

    def calculate_SCN_F(self, arch_objective=None):
        objective = np.asarray(self.lognormalize_objectives(arch_objective), dtype=np.float64)
        history = np.asarray(list(self.objectives_history), dtype=np.float64)
        #history = np.asarray(self.objectives_history, dtype=np.float64)
        n = len(objective)
        # Exclude self-comparison
        history = history[np.any(history != objective, axis=1)]
        # Calculate Euclidean distances (L2 norm)
        distances = np.linalg.norm(history - objective, axis=1)
        return np.min(distances)

    def calculate_SCN_G(self, vec = None, history = None, distance_type='None'):
        # Exclude self-comparison
        history = history[np.any(history != vec, axis=1)]
        print2(f'SCNG function {distance_type = }')
        print2(vec)
        print2(history)
        distances = get_distance(vector=vec, matrix=history, distance_type=distance_type)
        if distances is None or len(distances) == 0:
            print1(f'{Fore.RED} Warning: No valid distances calculated for SCN_G. Returning None.{Fore.RESET}')
            return None
        return np.min(distances)

    def get_vector_for_distances(self, arch=None):
        if self.INTEGER_ENCODING2:
            if self.encoding_type == 'BIN':
                vec = ''.join(arch.binary_encoding2_flattened)
            elif self.encoding_type == 'INT':
                vec = arch.integer_encoding2_flattened
            elif self.encoding_type == 'REAL':
                vec = arch.real_encoding2_flattened
            else:
                print(f"{Fore.RED} get_vector_for_distances Error: Unsupported encoding type {self.encoding_type}. Returning None.{Fore.RESET}")
        else:
            if self.encoding_type == 'BIN':
                vec = ''.join(arch.binary_encoding)
            elif self.encoding_type == 'INT':
                vec = arch.integer_encoding
            elif self.encoding_type == 'REAL':
                vec = arch.real_encoding
            else:
                print(f"{Fore.RED} get_vector_for_distances Error: Unsupported encoding type {self.encoding_type}. Returning None.{Fore.RESET}")
        
        if self.encoding_type == 'BIN':
            return np.fromiter((int(bit) for bit in vec), dtype=np.int8)
        return np.asarray(vec)

    def calculate_GD(self, archs_list=None, distance_type='None'):
        from itertools import combinations
        if archs_list is None or len(archs_list) < 2:
            return None
        add = 0.0
        N = 0
        print1(Fore.MAGENTA + f"Calculating GD for {len(archs_list)} architectures...")
        for a, b in combinations(archs_list, 2):
            vec_a = self.get_vector_for_distances(arch=a)
            vec_b = self.get_vector_for_distances(arch=b)
            distance = get_distance(vector=vec_a, matrix=np.asarray(vec_b).reshape(1, -1), distance_type=distance_type )
            if distance is None:
                return None
            add += distance[0]
            N += 1
        print1(Fore.MAGENTA + f"GD calculation complete. Total pairs: {N}")
        return add / N

    def calculate_GD2(self, archs_list = None, distance_type = 'None'):
        #Calculate the Genotype Distance (GD) of a list of architectures.
        #GD is the mean Hamming distance of all architectures in archs_list.
        from itertools import combinations
        add = 0
        N = 0
        print1(Fore.MAGENTA + f"Calculating GD for {len(archs_list)} architectures...")
        for a, b in combinations(archs_list, 2):
            #add += hamming_distance(a.integer_encoding, b.integer_encoding)
            add += get_distance(vectors_list = self.get_vectors_for_distances(arch1 = a, arch2 = b), distance_type = distance_type)
            N += 1
        print1(Fore.MAGENTA + f"GD calculation complete. Total pairs: {N}")
        GD = add/N
        #GD = GD/BLOCKS_CONSTANTS.SIZE_BLOCKGEN if ConfigClass.REPRESENTATION_TYPE == 'B' else GD/LAYERS_CONSTANTS.SIZE_LAYERGEN
        return GD

    def is_strictly_better(self, arch1, arch2):
        better_or_equal = ( arch1.acc >= arch2.acc and  arch1.flops <= arch2.flops and arch1.num_params <= arch2.num_params)
        strictly_better = ( arch1.acc >  arch2.acc or   arch1.flops < arch2.flops or   arch1.num_params < arch2.num_params)
        return better_or_equal and strictly_better


    def set_search_name(self):
        #self.search_name = f'{self.representation_type}{self.encoding_type}_'
        self.search_name = ''
        if config_tecnas.HHSE_GREEDY:
            self.search_name += f'GREEDY_{config_tecnas.HHSE_GREEDY_CRITERIA}'
        
        elif config_tecnas.HHSE_RANDOM:
            self.search_name += f'HRANDOM'
        elif config_tecnas.HHSE_TEC:
            self.search_name += f'HHSE_AOS'
        elif config_tecnas.NSGA2_NORMAL:
            nsga2_window_str = '_W' if config_tecnas.NSGA2_WINDOW else ''
            nsga2_window_size_str = f"{config_tecnas.NSGA2_WINDOW_SIZE_PERC*100:.0f}_" if config_tecnas.NSGA2_WINDOW else ''
            cross_name = f'{self.crossover_type}2' if self.INTEGER_ENCODING2 else self.crossover_type
            mut_name = f'{self.mutation_type}2' if self.INTEGER_ENCODING2 else self.mutation_type
            cross_name += f'M{self.HYPER_MUT_PROB*100:.0f}' if self.HYPER_MUTATION else ''
            mut_name += f'M{self.HYPER_MUT_PROB*100:.0f}' if self.HYPER_MUTATION else ''
            #hypermut = '_M' if self.HYPER_MUTATION else ''
            self.search_name += f'{cross_name}_{mut_name}{nsga2_window_str}{nsga2_window_size_str}'

    def validate_architecture(self, gen_list):
        # Validates and fixes the mutable CONV/POOL zone (indices 2-5).
        # Rules:
        #   1. At most 1 optional POOL in indices 2-5.
        #   2. No two consecutive POOLs anywhere in indices 2-6.
        #   3. Index 6 must always be a POOL (mandatory last POOL before FLATTEN).
        if ConfigClass.REPRESENTATION_TYPE != 'L':
            return gen_list  # No validation needed for block representation
        # ── Rule 3: enforce mandatory POOL at index 6 ───────────────────────
        if get_key_from_dict(gen_list[6]) not in ('POOLMAX', 'POOLAVG'):
            gen_list[6] = create_pool_max_layer()

        # ── Rule 1: at most 1 optional POOL in indices 2-5 ──────────────────
        pool_indexes = [i for i in range(2, 6) if get_key_from_dict(gen_list[i]) in ('POOLMAX', 'POOLAVG')]
        while len(pool_indexes) > 1:
            # Replace excess POOLs with a random CONV, keeping the first one
            idx_to_fix = pool_indexes.pop()  # remove last excess POOL
            gen_list[idx_to_fix] = create_conv_layer()

        # ── Rule 2: no consecutive POOLs in indices 2-6 ─────────────────────
        for i in range(2, 6):  # check pairs (i, i+1) up to (5, 6)
            if (get_key_from_dict(gen_list[i]) in ('POOLMAX', 'POOLAVG') and
                    get_key_from_dict(gen_list[i + 1]) in ('POOLMAX', 'POOLAVG')):
                gen_list[i] = create_conv_layer()

        return gen_list

    def log_out_of_bounds_arch(self, arch):
        statuses = []
        if arch.flops > self.MAXFLOPS:
            statuses.append("MAX FLOPS")
        if arch.flops < self.MINFLOPS:
            statuses.append("MIN FLOPS")
        if arch.num_params > self.MAXPARAMS:
            statuses.append("MAX PARAMS")
        if arch.num_params < self.MINPARAMS:
            statuses.append("MIN PARAMS")
        if not statuses:
            return
        # Join multiple violations if needed
        status_str = " | ".join(statuses)

        log_folder = os.path.join(self.pop[0].path_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)
        file_path = os.path.join(log_folder, "out_of_bounds_archs.csv")
        row = [str(arch.integer_encoding),  str(arch.genotype.gen_list), arch.flops, arch.num_params, status_str, self.MINFLOPS, self.MAXFLOPS,self.MINPARAMS, self.MAXPARAMS ]
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header only once
            if not file_exists:
                writer.writerow(["Integer_Encoding","Genotype", "FLOPs", "Num_Params", "Status", "Min_FLOPS", "Max_FLOPS", "Min_Params", "Max_Params"])
            writer.writerow(row)
        
    def reset_cross_mut_counts(self):
        self.succ_cross = 0
        self.succ_mut = 0
        self.total_cross = 0
        self.total_mut = 0

    def save_histories(self):
        self.accuracy_histories.append(self.pop[0].acc_hist)
        self.loss_histories.append(self.pop[0].loss_hist)
        self.best_acc_list.append(self.pop[0].acc)
        self.loss_list.append(self.pop[0].loss)
        self.cpu_hours_list.append(self.pop[0].cpu_hours)
        self.num_params_list.append(self.pop[0].num_params)
        self.flops_list.append(self.pop[0].flops)
        self.best_archs.append(self.pop[0])

    def report_op_counts(self, results_folder = ''):
        #For GREEDY_FIXED
        print(Fore.YELLOW + f"Reporting operator counts for execution {self.exec}...")
        folder_counts = os.path.join(results_folder, f'counts_{config_tecnas.HHSE_GREEDY_CRITERIA}')
        ensure_folder_exists(folder_counts)
        file_path = os.path.join(folder_counts, f"Op_count_Exec{self.exec}.csv" )
        df = pd.DataFrame(list(self.op_count.items()), columns=["Operator", "Count"])
        df.to_csv(file_path, index=False)
        print(Fore.YELLOW + f"Operator counts saved to {file_path}")
    def generate_total_counts(self, folder):
        all_dfs = []
        # Read all CSV files except total_counts.csv
        for file_name in os.listdir(folder):
            if (file_name.endswith(".csv") and file_name != "total_counts.csv"):
                file_path = os.path.join(folder, file_name)
                df = pd.read_csv(file_path)
                all_dfs.append(df)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        total_counts = combined_df.groupby("Operator", as_index=False)["Count"].sum().sort_values(by="Count", ascending=False)
        output_path = os.path.join(folder, "total_counts.csv")
        total_counts.to_csv(output_path, index=False)
        print(f"Total counts saved to: {output_path}")
        plt.figure(figsize=(12, 6))
        plt.bar(total_counts["Operator"], total_counts["Count"])
        plt.xlabel("Operator")
        plt.ylabel("Total Count")
        fixed_go = f'{self.crossover_type}_{self.mutation_type}' if not config_tecnas.HHSE_GREEDY_FIXED_RANDOM else "RANDOM GO"
        title = f"Total Operator Counts {config_tecnas.HHSE_GREEDY_CRITERIA} Fixed GO: {fixed_go} - All Executions"
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        output_plot = os.path.join(folder, f"counts_barplot_{config_tecnas.HHSE_GREEDY_CRITERIA}.png")
        plt.savefig(output_plot)
        plt.close()
        print(f"Bar plot saved to: {output_plot}")

    def compute_topk(self, model, x_test, y_test, k=5):
            probs = model.predict(x_test, verbose=0)
            top_k_preds = np.argsort(probs, axis=1)[:, -k:]
            correct = 0
            for i in range(len(y_test)):
                if y_test[i] in top_k_preds[i]:
                    correct += 1
            return correct / len(y_test)

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

    def flops_params_gen_list_str(self, get_realMetrics = True, gen_list_str = None):
        #Returns FLOPs and NumParams for a given gen_lis
        arch_obj = self.create_arch_from_genlist(ast.literal_eval(gen_list_str), create_model = get_realMetrics)
        flops =  calculate_model_flops(arch_obj.model) if get_realMetrics else count_flops_analytic(gen_list_str, input_res=32) 
        num_params, sizeMB = calculate_model_params(arch_obj.model) if get_realMetrics else count_params_analytic(gen_list_str)
        print2(Fore.RED + f"FLOPs Params calculation complete")
        return flops, num_params, sizeMB

    def create_arch_from_genlist(self, gen_list = '', create_model = True):
        #Create an architecture from a gen_list
        arch_obj = LayerRepresentation(genotypeObj=Genotype(gen_list=gen_list)) if ConfigClass.REPRESENTATION_TYPE == 'L' else BlockRepresentation(genotypeObj=Genotype(gen_list=gen_list), INTEGER_ENCODING2 = self.INTEGER_ENCODING2)
        arch_obj.model = self.create_model(arch_obj) if create_model else None
        return arch_obj

    def flops_params_folder(self, folder_path = '', realMetrics = True):
        ConfigClass.IMPORT_TENSORFLOW if realMetrics else None
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                print(f'Processing file: {filename} for FLOPs and number of params calculation')
                filepath = os.path.join(folder_path, filename)
                df = pd.read_csv(filepath, encoding='utf-8')
                target_rows = df[(df['FLOPs'].isna())]
                total = len(target_rows)
                arch_count = 0
                for idx in target_rows.index:
                    arch_count += 1
                    arch_ID = df.at[idx, 'ID']
                    print(Fore.LIGHTBLUE_EX + f'Processing arch {arch_ID}:  {arch_count}/{total} ============== ')
                    genotype = df.at[idx, 'Genotype']
                    flops, num_params, sizeMB = self.flops_params_gen_list_str(get_realMetrics = realMetrics, gen_list_str = str(genotype))
                    df.at[idx, 'FLOPs'] = flops
                    df.at[idx, 'Num_Params'] = num_params
                    df.at[idx, 'SizeMB'] = sizeMB
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
            self.log_out_of_bounds_arch(arch)
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


    def selection(self, pop = None):
        print2('\nSelecting best parents and children for the new population')
        #Based on the SORT_ARCHS variable, select the best architectures from the sorted pool.
        if self.NSGA2 == False:
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
            pop = self.sorted_pool[begin:end]
        else: #Use NSGA-II selection
            self.nsga2_obj = TECNAS_NSGAII(pop = self.pool, pop_size = self.NPOP, tecnasObj = self)
            pop, self.HV, self.riesz_energy = self.nsga2_obj.select_and_HV(self.pool)
            
            if self.generation == 1:
                self.dHV = 0
            else:
                self.dHV = self.HV - self.prev_HV
            self.prev_HV = self.HV

            pop = sorted(pop, key=lambda obj: obj.acc, reverse = True) #Best archs first
            print('Selection with NSGA-II complete')
        print2('\nSelecting best parents and children for the new population complete')
        return pop

    def apply_all_GOs(self): #==================================================APPLY ALL GOs=======================================================
     
        def determine_GOs(self, GO_SELECTED = False, GO_SELECTED_NAME = ''):
            #Use best SELECTED identified GOs for each stage (CROSS, MUT, HD, DHV) for GREEDY_COMBINED. set_GREEDY_GOs_list returns 4 lists with all GOs (SPC_MPAR, TPC_MTYP, etc)
            #GO_SELECTED: CROSS_SELECTED, MUT_SELECTED, etc.
            GOs_CROSS, GOs_MUT, GOs_HV, GOs_GD = config_tecnas.set_GREEDY_GOs_list(self.representation_type, self.encoding_type)
            GOs_dict = {'CROSS': GOs_CROSS, 'MUT': GOs_MUT, 'HV': GOs_HV, 'GD': GOs_GD}
            if GO_SELECTED:
                return GOs_dict[GO_SELECTED_NAME]
                       
            progress = self.generation / self.total_gen
            if progress <= self.limits[0]: #CROSS
                return GOs_CROSS
            elif progress <= self.limits[1]:
                return GOs_MUT
            elif progress <= self.limits[2]:
                return GOs_HV
            elif progress <= self.limits[3]:
                return GOs_GD
            else:
                print(Fore.LIGHTRED_EX + "No GOs selected for this stage. This should not happen if limits are set correctly.")
                return None, None
        def determine_fixed_GOs():
            #If GREEDY_FIXED, return the first element of the list returned by set_GREEDY_FIXED_GOs_list.
            #If GREEDY_FIXED_RANDOM, return the best go.
            if config_tecnas.HHSE_GREEDY_FIXED_RANDOM:
                cross_mut_str = f'{random.choice(config_tecnas.crossoverList_hhse)}_{random.choice(config_tecnas.mutationList_hhse)}'
                while cross_mut_str == 'NONE_NONE' and config_tecnas.IGNORE_RANDOM:
                    cross_mut_str = f'{random.choice(config_tecnas.crossoverList_hhse)}_{random.choice(config_tecnas.mutationList_hhse)}'
                return cross_mut_str
            GOs_CROSS, GOs_MUT, GOs_HV, GOs_GD = config_tecnas.set_GREEDY_FIXED_GOs_list(self.representation_type, self.encoding_type)
            GOs_dict = {'CROSS': GOs_CROSS, 'MUT': GOs_MUT, 'HV': GOs_HV, 'GD': GOs_GD}
            return GOs_dict[config_tecnas.HHSE_GREEDY_CRITERIA][0]
        #Apply all genetic operators. Save them as a list of objects.
        self.report_training = False #Disable training report for this function
        self.pool = []
        self.children = []
        self.GOs_population_dict = {}
        self.generation_status = {}
        GO_SELECTED = True if 'SELECTED' in config_tecnas.HHSE_GREEDY_CRITERIA else False
        GO_SELECTED_NAME = config_tecnas.HHSE_GREEDY_CRITERIA.split('_')[0] if GO_SELECTED else None   #CROSS, from CROSS_SELECTED, etc.
        #self.GOs_names = [f"{cross}_{mut}" for cross, mut in product(crossover_list, mutation_list)]
        self.GOs_names = [f"{cross}_{mut}" for cross, mut in product(self.crossover_types_list, self.mutation_types_list)] if not GO_SELECTED else determine_GOs(self, GO_SELECTED = GO_SELECTED, GO_SELECTED_NAME = GO_SELECTED_NAME)
        if config_tecnas.IGNORE_RANDOM and 'NONE_NONE' in self.GOs_names:
            self.GOs_names.remove('NONE_NONE')
        for GO in self.GOs_names:
            self.crossover_type, self.mutation_type = GO.split('_')
            if config_tecnas.IGNORE_RANDOM and self.crossover_type == 'NONE' and self.mutation_type == 'NONE':
                continue
            self.selected_by = ''
            print2(f'{self.crossover_type}_{self.mutation_type} COMBINATION')
            self.genetic_operators()
            cross_mut_str = f'{self.crossover_type}_{self.mutation_type}'
            self.report_training = False
            self.pool = self.pop + self.children
            self.sorted_pool = sorted(self.pool, key=lambda obj: obj.acc, reverse = True) #Best archs first
            print2('Applying all genetic operators in course. Selecting next generation')
            self.GOs_population_dict[cross_mut_str] = copy.deepcopy(self.selection(pop = self.pool))
            self.GD = self.calculate_GD(archs_list = self.GOs_population_dict[cross_mut_str], distance_type = self.distance_type) if config_tecnas.HHSE_GREEDY_FIXED else 0
            print2('Selection of next generation complete\n')
            #self.GOs_HV[cross_mut_str] = [HV_actual, HV_prev, dHV]
            self.GOs_HV[cross_mut_str][2] = 0 if self.generation == 1 else self.HV - self.GOs_HV[cross_mut_str][0] #Calculate dHV
            self.GOs_HV[cross_mut_str] =  [self.HV, self.GOs_HV[cross_mut_str][0], self.GOs_HV[cross_mut_str][2]] 

            statusObj = Status(tecnasObj = self, crosstype = self.crossover_type, muttype = self.mutation_type, pop = self.GOs_population_dict[cross_mut_str])
            self.reporter.save_generation_status(statusObj) if not config_tecnas.HHSE_GREEDY else None
            self.generation_status[cross_mut_str] = statusObj
                
        best_go = self.select_best_GOs(criteria = config_tecnas.HHSE_GREEDY_CRITERIA if not GO_SELECTED else GO_SELECTED_NAME)
        #Update the information of the status for the best_go. This includes the history (criteria that passed) and why it was selected (selected_by)
        self.op_count[best_go] = self.op_count.get(best_go, 0) + 1
        best_go = best_go if not config_tecnas.HHSE_GREEDY_FIXED else determine_fixed_GOs()
        self.generation_status[best_go].update_chosen_GO(self.GOs_history, self.op_count)
        self.reporter.save_generation_status(self.generation_status[best_go])
        for GO in self.GOs_HV:
            self.GOs_HV[GO] = self.GOs_HV[best_go].copy() #Update previous HV for all GOs for the next generation. This is needed for the dHV calculation in the next generations.
        return best_go, self.GOs_population_dict[best_go]

    def select_best_GOs(self, criteria = ''): #==================================================SELECT BEST GOs=======================================================
        
        combined_str = ''
        if criteria == 'COMBINED' or criteria == 'COMBINED_SELECTED':
            combined_str = 'COMBINED_'
            progress = self.generation / self.total_gen
            if progress <= self.limits[0]:
                self.markov_type = config_tecnas.HHSE_MARKOV_TYPES[0]
                best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].succ_cross)
            elif progress <= self.limits[1]:
                self.markov_type = config_tecnas.HHSE_MARKOV_TYPES[1]
                best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].succ_mut)
            elif progress <= self.limits[2]:
                self.markov_type = config_tecnas.HHSE_MARKOV_TYPES[2]
                best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].HV)
            elif progress <= self.limits[3]:
                self.markov_type = config_tecnas.HHSE_MARKOV_TYPES[3]
                #best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].dHV)
                best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].GD)
            else:
                print(Fore.LIGHTRED_EX + "No criteria selected for this stage. This should not happen if limits are set correctly. {progress = }")
                best_go = None
        elif criteria == 'HV':
            best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].HV)
        elif criteria == 'DHV':
            best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].dHV)
        elif criteria == 'MUT':
            best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].succ_mut)
        elif criteria == 'CROSS':
            best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].succ_cross)
        elif criteria == 'ACC':
            best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].mean_accuracy)
        elif criteria == 'GD':
            best_go = max(self.GOs_names, key = lambda go: self.generation_status[go].GD)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {criteria = } is not a valid criteria for selecting the best GO. No GO will be selected.')
            best_go = None
        
        self.GOs_history.append(best_go)  
        self.selected_by = f'{combined_str}{self.markov_type}' if criteria == 'COMBINED' else criteria
        return best_go

    def genetic_operators(self):
        print2(Fore.LIGHTYELLOW_EX + 'Generating offspring')
        self.reset_cross_mut_counts()
        if self.search_strategy == 'GA':
            #================================================= CROSSOVER ==========================================================================================
            if self.crossover_type != 'NONE':
                print2(Fore.GREEN + f'\nUsing Crossover {self.crossover_type}')
                self.generate_offspring() #CROSSOVER
                print2(Fore.GREEN + 'Crossover complete\n')
            else: #NO CROSSOVER
                #Copy the parents to the children in case of only mutation. (Mutation only works with children and they are generated in the crossover function)
                print2(Fore.RED + 'No Crossover operation selected. Copying parents to children')
                self.children = []
                for arch in self.pop:
                    child = copy.deepcopy(arch)
                    child = self.make_child(child, arch, arch, distance_type = self.distance_type)
                    self.children.append(child) #Make a child of itself. This is to keep the same structure as the crossover function. Parents are itself
                    print2(Fore.RED + 'Copying parents to children complete\n')
            #================================================= MUTATION ==========================================================================================
            if self.mutation_type != 'NONE': 
                print2(Fore.GREEN + '\nMutating offspring')
                self.mutate_children(self.mutation_type)
                print2(Fore.GREEN + 'Mutating children done\n')
            else: #NO MUTATION
                print2(Fore.RED + 'No Mutation operation selected. Children will not be mutated\n')
        else: #RANDOM SEARCH
            print2(Fore.GREEN + f'\nGenerating random children {self.search_strategy}')
            self.children = []
            for i in range(self.NPOP):
                random_arch = self.random_individual()
                random_arch = self.train_model(random_arch)
                self.children.append(random_arch)
            print2('Generating and training initial population complete')
            print2(Fore.GREEN + 'Generating random children\n')
        print2(Fore.LIGHTYELLOW_EX + 'Generating offspring complete\n')

    def child_better_parents(self, child, parent1, parent2):
        if self.is_strictly_better(child, parent1) or self.is_strictly_better(child, parent2):
            return True
        else:
            return False
        
    def get_normalize_dataset(self):
        # ------------------ LOAD DATASET ------------------
        if ConfigClass.DATASET == 'CIFAR10':
            print(Fore.YELLOW + f'Loading CIFAR10 dataset' + Fore.RESET)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif ConfigClass.DATASET == 'CIFAR100':
            print(Fore.YELLOW + f'Loading CIFAR100 dataset' + Fore.RESET)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {ConfigClass.DATASET} dataset not recognized.' + Fore.RESET)
            return
        y_train = y_train.flatten()
        if ConfigClass.DATASET_PART == 1:
            subset_x_train = x_train
            subset_y_train = y_train
        else:
            subset_x_train, _, subset_y_train, _ = train_test_split(x_train, y_train, train_size=ConfigClass.DATASET_PART, stratify=y_train, random_state=42)
        subset_y_train = subset_y_train.reshape(-1, 1)
        x_train_part, x_val, y_train_part, y_val = train_test_split(subset_x_train, subset_y_train, test_size=0.1, stratify=subset_y_train, random_state=42)
        self.x_train = x_train_part / 255.0
        self.x_val   = x_val        / 255.0
        self.x_test  = x_test       / 255.0
        self.y_train = y_train_part.flatten()
        self.y_val   = y_val.flatten()
        self.y_test  = y_test.flatten()
        self.train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        self.train_datagen.fit(self.x_train) 
        self.train_generator = self.train_datagen.flow(self.x_train, self.y_train, batch_size=Globals.BATCH_SIZE)

        # VALIDATION GENERATOR (NO AUGMENTATION)
        self.validation_datagen = ImageDataGenerator()
        self.validation_generator = self.validation_datagen.flow(self.x_val, self.y_val, batch_size=Globals.BATCH_SIZE)


    def make_child(self, child=None, parent1=None, parent2=None, distance_type='None'):
        child.isChild = True
        child.isMutant = False
        child.P1Idx = parent1.idx
        child.P2Idx = parent2.idx
        child.P1IntegerEncoding = parent1.integer_encoding
        child.P2IntegerEncoding = parent2.integer_encoding
        child.before_mutationIdx = ''
        child.before_mutationIntegerEndcoding = None
        print2(f"\nmake_child function:\n{child.integer_encoding = }\n{parent1.integer_encoding = }\n{parent2.integer_encoding = }\n")
        # Get the appropriate integer encodings
        vec_child = self.get_vector_for_distances(arch=child)
        vec_parent1 = self.get_vector_for_distances(arch=parent1)
        vec_parent2 = self.get_vector_for_distances(arch=parent2)
        # Calculate distances
        distance_p1 = get_distance(vector=vec_child, matrix=np.asarray(vec_parent1).reshape(1, -1), distance_type=distance_type)
        distance_p2 = get_distance(vector=vec_child, matrix=np.asarray(vec_parent2).reshape(1, -1), distance_type=distance_type)
        child.dP1 = distance_p1[0] if distance_p1 is not None else None
        child.dP2 = distance_p2[0] if distance_p2 is not None else None
        print2('Distance calculation done')
        child.dBM = 0
        child.P1acc = parent1.acc
        child.P2acc = parent2.acc
        child.P1flops = parent1.flops
        child.P2flops = parent2.flops
        child.P1params = parent1.num_params
        child.P2params = parent2.num_params
        return child

    def make_mutant(self, archM=None, archOriginal=None, distance_type='None'):
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
        print2(f"\nmake_mutant function:\n{archM.integer_encoding = }\n{archOriginal.integer_encoding = }\n")
        # Get the appropriate integer encodings
        vec_mutant = self.get_vector_for_distances(arch=archM)
        vec_original = self.get_vector_for_distances(arch=archOriginal)
        # Calculate distance
        distance = get_distance(vector=vec_mutant, matrix=np.asarray(vec_original).reshape(1, -1), distance_type=distance_type)
        archM.dBM = distance[0] if distance is not None else None
        archM.P1acc = archOriginal.P1acc
        archM.P2acc = archOriginal.P2acc
        archM.P1flops = archOriginal.P1flops
        archM.P2flops = archOriginal.P2flops
        archM.P1params = archOriginal.P1params
        archM.P2params = archOriginal.P2params
        archM.set_genoStr()

        return archM

    def make_parent(self, arch):
        #Keep the information of the parent architecture
        #ind = copy.deepcopy(arch)
        arch.isChild = False
        arch.isMutant = False
        '''
        arch.P1Idx = ''
        arch.P2Idx = ''
        arch.P1IntegerEncoding = None
        arch.P2IntegerEncoding = None
        arch.before_mutationIdx = ''
        arch.before_mutationIntegerEndcoding = None
        arch.dP1 = -1
        arch.dP2 = -1
        arch.dBM = -1
        '''
        return arch

    def change_block(self, original_arch = None,  mutation_type = None, INTEGER_ENCODING2 = False):
        mutator_obj = Mutator()
        arch = copy.deepcopy(original_arch)
        real_vector = None
        print2(f'change block function before mutation: {arch.integer_encoding = }')
        mutable_block_indexes = list(range(len(arch.integer_encoding))) if self.INTEGER_ENCODING2 else BLOCKS_CONSTANTS.MUTABLE_BCHANGEPARAM_INDEXES
        if mutation_type == 'MBFLIP':
            vec_bin = arch.binary_encoding2_flattened if INTEGER_ENCODING2 else arch.binary_encoding 
            vec_bin = mutator_obj.mutate_bitflip(binary_encoding = vec_bin, MAX_INT = len(BlockRepresentation.all_blocks)-1, INTEGER_ENCODING2 = INTEGER_ENCODING2, MUT_PROB = self.MUT_PROB)
            vec_int = BlockRepresentation.binary_encoding_to_integer_encoding(binary_encoding = vec_bin)
            vec_int = BlockRepresentation.flattened_integer_encoding2_to_integer_encoding(vec_int) if INTEGER_ENCODING2 else vec_int
        elif mutation_type == 'MPAR': 
            vec_int = arch.integer_encoding2_flattened if INTEGER_ENCODING2 else arch.integer_encoding
            vec_int = mutator_obj.mutate_block_parameters(vec_int = vec_int, INTEGER_ENCODING2 = INTEGER_ENCODING2, MUT_PROB = self.MUT_PROB)
            vec_int = BlockRepresentation.flattened_integer_encoding2_to_integer_encoding(vec_int) if INTEGER_ENCODING2 else vec_int
        elif mutation_type == 'MSWAP':
            vec_int = arch.integer_encoding2_flattened if INTEGER_ENCODING2 else arch.integer_encoding
            vec_int = mutator_obj.mutate_block_swap(vec_int = vec_int, INTEGER_ENCODING2 = INTEGER_ENCODING2, MUT_PROB = self.MUT_PROB)
            vec_int = BlockRepresentation.flattened_integer_encoding2_to_integer_encoding(vec_int) if INTEGER_ENCODING2 else vec_int
        elif mutation_type == 'MPOLY':
            mpoly_indexes = range(len(BlockRepresentation().all_blocks)) #0 1 2 3... 63 polynomial_mutation function deals with max and min for integer_encoding2.
            vreal = arch.real_encoding2_flattened if INTEGER_ENCODING2 else arch.real_encoding 
            real_vector = mutator_obj.polynomial_mutation(vreal = vreal, t = self.generation, ylow = min(mpoly_indexes), yhigh = max(mpoly_indexes), INTEGER_ENCODING2 = INTEGER_ENCODING2, MUT_PROB = self.MUT_PROB)
            vec_int = BlockRepresentation.flattened_real_encoding2_to_integer_encoding(real_vector) if INTEGER_ENCODING2 else BlockRepresentation.real_encoding_to_integer_encoding(real_vector)
        elif mutation_type == 'MNUF':
            mnounf_indexes = range(len(BlockRepresentation().all_blocks)) #0 1 2 3... 63 nouniform_mutation function deals with max and min for integer_encoding2.
            vreal = arch.real_encoding2_flattened  if INTEGER_ENCODING2 else arch.real_encoding 
            real_vector = mutator_obj.nouniform_mutation(vreal = vreal, t = self.generation, ylow = min(mnounf_indexes), yhigh = max(mnounf_indexes), INTEGER_ENCODING2 = INTEGER_ENCODING2, MUT_PROB = self.MUT_PROB)
            vec_int = BlockRepresentation.flattened_real_encoding2_to_integer_encoding(real_vector) if INTEGER_ENCODING2 else BlockRepresentation.real_encoding_to_integer_encoding(real_vector)
        else:
            print(f'ERROR: {mutation_type} Mutation type not recognized for {self.representation_type}{self.encoding_type}')
            return None

        arch.genotype.gen_list = arch.integer_vector_to_genList(vec_int)
        arch.genotype.gen_list = BlockRepresentation.update_genlist(arch.genotype.gen_list)
        arch.integer_encoding, arch.integer_encoding2, arch.integer_encoding2_flattened, arch.real_encoding, arch.real_encoding2, arch.real_encoding2_flattened, arch.binary_encoding, arch.binary_encoding2_flattened = BlockRepresentation.update_vector_encodings(int_vec = vec_int, INTEGER_ENCODING2 = self.INTEGER_ENCODING2, binary_encoding = True if self.encoding_type == 'BIN' else False)
        arch.set_genotype(arch.genotype)
        print2(f'change block function after mutation: {arch.integer_encoding = }')
        return arch

        
    

    def mutate_layer(self, child, mutation_type):
        #gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
        #Mutate each layer with probability MUT_PROB.
        #Save it in the genotype and update it.
        #Return the mutated architecture
        mutator_obj = Mutator()
        archMut = LayerRepresentation(genotypeObj = None)
        archMut.genotype = copy.deepcopy(child.genotype)
                
        if child.encoding == 'INT':
            if mutation_type == 'MPAR':
                layers_indexes = LAYERS_CONSTANTS.MUTABLE_LCHANGEPARAM_INDEXES #Only this indexes may change their parameters
                mutation_function = mutator_obj.mutate_layer_parameters
            elif mutation_type == 'MSWAP':
                layers_indexes = LAYERS_CONSTANTS.MUTABLE_LSWAP_INDEXES #Only this indexes may change type
                mutation_function = mutator_obj.mutate_layer_swap
            else:
                print(Fore.RED + f'ERROR: {mutation_type} Mutation type not recognized for {self.representation_type}{self.encoding_type}' + Fore.RESET)
                return None
            for layer_idx in layers_indexes:
                if random.random() < self.MUT_PROB:
                    archMut.genotype.gen_list = mutation_function(archMut.genotype, layer_idx)
                    archMut.set_genotype(archMut.genotype)

        elif child.encoding == 'BIN':
            MAX_INT = self.MAXINT
            if mutation_type == 'MBFLIP':
                archMut.binary_encoding = mutator_obj.mutate_bitflip(child.binary_encoding, MAX_INT)
                archMut.binary_encoding_to_integer_encoding()
                archMut.genotype.gen_list = archMut.integer_vector_to_genList(archMut.integer_encoding)
                archMut.set_genotype(archMut.genotype)
            else:
                print(Fore.RED + f'ERROR: {mutation_type} Mutation type not recognized for {self.representation_type}{self.encoding_type}' + Fore.RESET)
                return None

        return archMut

   
    def mutate_children(self, mutation_type):
        if self.search_strategy == 'RANDOM':
            return
        for i in range(len(self.children)):
            name_before_mut = self.children[i].idx
            print2(Fore.GREEN + f'MUTATING {self.children[i].idx} with {mutation_type} mutation')
            if ConfigClass.REPRESENTATION_TYPE == 'L':
                mutated_child = self.mutate_layer(self.children[i], mutation_type)
                mutated_child.genotype.gen_list = self.validate_architecture(mutated_child.genotype.gen_list)
                mutated_child.set_genoStr()
                
            elif ConfigClass.REPRESENTATION_TYPE == 'B':
                mutated_child = self.change_block(original_arch = self.children[i], mutation_type = mutation_type, INTEGER_ENCODING2 = self.INTEGER_ENCODING2)

            else:
                print(Fore.LIGHTRED_EX + f'ERROR (mutate_children): {self.representation_type} Representation type not recognized' + Fore.RESET)
                return None
            print2(Fore.GREEN + f'MUTATING {self.children[i].idx} with {mutation_type} mutation complete\n')
            self.total_mut += 1
            mutated_child.idx = str(name_before_mut) + '[M]' #Add M to the index to indicate it was mutated      
            mutated_child = self.make_mutant(mutated_child, self.children[i], distance_type = self.distance_type)
            mutated_child = self.train_model(mutated_child)
            print2('Assesing if the mutation was succesful')
            #self.succ_mut += 1 if mutated_child.acc > self.children[i].acc else 0
            self.succ_mut += 1 if self.is_strictly_better(mutated_child, self.children[i]) else 0
            print2('Assesing if the mutation was succesful completed')
            self.children[i] = mutated_child

    def crossover(self, arch_obj1 = None, arch_obj2 = None):
        #For integer_encoding2, remember that arch_integer_encoding looks like [0,1,0,3,0,1] but crossover works with [(0,1),(0,3),(0,1)] and returns the same format.
        crossover_obj = Crossover()
        if self.INTEGER_ENCODING2: #Crossover works with [(0,1),(0,3),(0,1)] except SBX.
            v1 = arch_obj1.binary_encoding2_flattened if self.encoding_type == 'BIN' else arch_obj1.integer_encoding2
            v2 = arch_obj2.binary_encoding2_flattened if self.encoding_type == 'BIN' else arch_obj2.integer_encoding2
        else:
            v1 = arch_obj1.binary_encoding if self.encoding_type == 'BIN' else arch_obj1.integer_encoding
            v2 = arch_obj2.binary_encoding if self.encoding_type == 'BIN' else arch_obj2.integer_encoding

        if self.crossover_type == 'SPC':
            point = random.choice(range(len(v1))) if self.encoding_type == 'BIN' else random.choice(TECNAS.REP_CONSTANTS.SPC_INDEXES)
            child1_vec, child2_vec = crossover_obj.single_point_crossover(v1 = v1, v2 = v2, point = point)
        elif self.crossover_type == 'TPC':
            [point1, point2] = random.sample(range(len(v1)), 2) if self.encoding_type == 'BIN' else random.sample(TECNAS.REP_CONSTANTS.SPC_INDEXES, 2)
            child1_vec, child2_vec = crossover_obj.two_point_crossover(v1 = v1, v2 = v2, point1 = point1, point2 = point2)
        elif self.crossover_type == 'UC':
            child1_vec, child2_vec = crossover_obj.uniform_crossover(v1 = v1, v2 = v2, crossover_indexes = range(len(v1)) if self.encoding_type == 'BIN' else TECNAS.REP_CONSTANTS.SPC_INDEXES)
        elif self.crossover_type == 'SBX':
                #INTEGER_ENCODING2 is dealt inside SBX. Specifically for min and max ints.
                child1_vecReal = arch_obj1.real_encoding2_flattened if self.INTEGER_ENCODING2 else arch_obj1.real_encoding
                child2_vecReal = arch_obj2.real_encoding2_flattened if self.INTEGER_ENCODING2 else arch_obj2.real_encoding
                child1_vecReal, child2_vecReal = crossover_obj.SBX(v1_real = child1_vecReal, v2_real = child2_vecReal, eta_c = 2.0, min_int = 0, max_int = 63, INTEGER_ENCODING2 = self.INTEGER_ENCODING2)
                #Check if INTEGER_ENCODING2 is True. If it is, go from flatened_real encoding to integer_encoding.
                child1_vec = BlockRepresentation.flattened_real_encoding2_to_integer_encoding2(child1_vecReal) if self.INTEGER_ENCODING2 else BlockRepresentation.real_encoding_to_integer_encoding(child1_vecReal)
                child2_vec = BlockRepresentation.flattened_real_encoding2_to_integer_encoding2(child2_vecReal) if self.INTEGER_ENCODING2 else BlockRepresentation.real_encoding_to_integer_encoding(child2_vecReal)
        else:
            print(f'UNRECOGNIZED CROSSOVER METHOD {self.crossover_type}')
            return None, None

        if self.encoding_type == 'BIN':
            #print(f'tecnas crossover \nv1 = {v1}\nv2 = {v2}\n')
            #print(f'child1_vec = {child1_vec}\nchild2_vec = {child2_vec}\n')
            #print(f'arch_obj1.int = {arch_obj1.integer_encoding}\narch_obj2.int = {arch_obj2.integer_encoding}\n')
            child1_vec = BlockRepresentation.binary_encoding_to_integer_encoding(binary_encoding = child1_vec)
            child2_vec = BlockRepresentation.binary_encoding_to_integer_encoding(binary_encoding = child2_vec)
            child1_vec = BlockRepresentation.flattened_to_tuples(child1_vec) if self.INTEGER_ENCODING2 else child1_vec
            child2_vec = BlockRepresentation.flattened_to_tuples(child2_vec) if self.INTEGER_ENCODING2 else child2_vec
            #print(f'tecnas crossover \nchild1_vec = {child1_vec}\nchild2_vec = {child2_vec}')

        child1_intvec = BlockRepresentation.integer_encoding2_to_integer_encoding(child1_vec) if self.INTEGER_ENCODING2 else child1_vec
        child2_intvec = BlockRepresentation.integer_encoding2_to_integer_encoding(child2_vec) if self.INTEGER_ENCODING2 else child2_vec

        child1_arch = self.vector_to_arch(intvec = child1_intvec, arch_obj1_idx = arch_obj1.idx, arch_obj2_idx = arch_obj2.idx, is_first_child = True)
        child2_arch = self.vector_to_arch(intvec = child2_intvec, arch_obj1_idx = arch_obj1.idx, arch_obj2_idx = arch_obj2.idx, is_first_child = False)    
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
        print2(f'GA SEARCH STRATEGY')
        self.nsga2_obj = TECNAS_NSGAII(pop = self.pop, pop_size = self.NPOP, tecnasObj = self) if self.NSGA2 == True else None
        nsga2_parents_list = self.nsga2_obj.select_parent_pairs() if self.NSGA2 == True else None
        
        for i in range(self.NPOP//2): #//2 because you add two children
            print2('\nSelecting parents')
            parent1, parent2 = self.random_parent_selection() if self.NSGA2 == False else nsga2_parents_list[i]
            print2(Fore.LIGHTBLUE_EX + f'Using crossover')
            child1, child2 = self.crossover(arch_obj1=parent1, arch_obj2=parent2)
            print2(f'Generate offspring function:\n{parent1.idx = }{parent1.integer_encoding = }\n{parent2.idx = }{parent2.integer_encoding = }\n{child1.idx = }{child1.integer_encoding = }\n{child2.idx = }{child2.integer_encoding = }\n')
            #child1.genotype.gen_list = self.validate_architecture(child1.genotype.gen_list)
            #child2.genotype.gen_list = self.validate_architecture(child2.genotype.gen_list)
            child1.set_genoStr()
            child2.set_genoStr()
            child1 = self.make_child(child1, parent1, parent2, distance_type = self.distance_type)
            child2 = self.make_child(child2, parent1, parent2, distance_type = self.distance_type)
            self.total_cross += 2
            print2('\nTraining children')
            child1 = self.train_model(child1)
            child2 = self.train_model(child2)
            print2('Children training done')
            print2('\nEvaluating if crossover was succesful')
            self.succ_cross += 1 if self.child_better_parents(child1, parent1, parent2) else 0
            self.succ_cross += 1 if self.child_better_parents(child2, parent1, parent2) else 0
            print2('Evaluation done\n')
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
    
    def random_genlist(self):
        #     0                 1                  2                     3                      4                  5                    6              7                      8                   9               10
        #[{'INP': 32}, {'CONV': [128, 5]}, {'CONV': [128, 5]}, {'POOLMAX': [-1, 3]}, {'CONV': [64, 3]}, {'CONV': [256, 5]}, {'POOLMAX': [-1, 2]}, {'FLATTEN': None}, {'DENSE': [512, 'relu']}, {'LAST_DENSE': [10, 'softmax']}]
        #[5, 5, 9, 2, 7, 8, 12]
        #[C, C, P, C, C, P, D]
        gen_list = []
        if self.representation_type == 'L':
            gen_list = []
            pool_count = 0
            gen_list.append({'INP': Globals.INPUT_SIZE})
            gen_list.append(create_conv_layer())

            # ── Mutable middle slots ─────────────────────────────────────────────
            # SIZE_GENLIST - NUM_FIXED_LAYERS gives the total number of mutable
            # slots. We've already filled one (the mandatory first CONV), so we
            # iterate over the remaining ones.
            # We stop 2 slots early: one already filled (first CONV), one reserved
            # for the mandatory POOL that must appear right before FLATTEN.
            num_mutable = LAYERS_CONSTANTS.SIZE_GENLIST - LAYERS_CONSTANTS.NUM_FIXED_LAYERS
            for i in range(num_mutable - 2):          # -2: first slot filled + last slot reserved for POOL
                prev_is_pool = get_key_from_dict(gen_list[-1]) in ('POOLMAX', 'POOLAVG')
                # Reserve one pool budget for the mandatory last slot
                can_add_pool = (pool_count < LAYERS_CONSTANTS.MAX_POOLS - 1 and not prev_is_pool)
                if can_add_pool and random.random() < 0.3:   # 30 % chance to insert a pool
                    gen_list.append(create_pool_max_layer())
                    pool_count += 1
                else:
                    gen_list.append(create_conv_layer())

            # ── Mandatory POOL right before FLATTEN ──────────────────────────────
            gen_list.append(create_pool_max_layer())

            gen_list.append({'FLATTEN': None})
            gen_list.append(create_dense_layer())
            gen_list.append(create_dense_layer(last_dense=True))
            return gen_list
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
        genotype = Genotype(self.representation_type, 'INT', gen_list)

        if self.representation_type == 'L':
            arch_obj = LayerRepresentation(idx = str(idx), genotypeObj=genotype)
        elif self.representation_type == 'B':
            arch_obj = BlockRepresentation(idx = str(idx), genotypeObj=genotype, INTEGER_ENCODING2 = self.INTEGER_ENCODING2)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR (TECNAS.create_individual): {self.representation_type} Representation type not recognized' + Fore.RESET)
            return None
        
        return arch_obj

    def random_individual(self):
        gen_list = []
        gen_list = self.random_genlist()
        genotype = Genotype(self.representation_type, self.encoding_type, gen_list)
        idx = random.choice(ARCH_NAMES_LIST)

        if self.representation_type == 'L':
            arch_obj = LayerRepresentation(idx = str(idx), genotypeObj = genotype)
        elif self.representation_type == 'B':
            arch_obj = BlockRepresentation(idx = str(idx), genotypeObj = genotype, INTEGER_ENCODING2 = self.INTEGER_ENCODING2)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: {self.representation_type} Representation type not recognized' + Fore.RESET)
            return None
        return arch_obj

    def simulate_training(self, arch, report = True):
        print2(f"Simulating trainining model {arch.idx}")
        # Simulate training and return the architecture with simulated accuracy and loss
        arch.acc = random.randint(0, 100)
        arch.loss = random.uniform(0, 1)
        arch.flops = random.randint(0, 100)
        arch.cpu_hours = random.uniform(0, 1)
        arch.num_params = random.randint(0, 100)
        arch.gen_creation = self.generation
        if report == True:
            self.reporter.save_arch_info(self, arch)
        
        print2(Fore.GREEN + f'Simulating Training {arch.idx} complete. Accuracy {arch.acc}\n')

    def training_with_surrogate(self, arch, report = True):
        print2(f"Predicting model {arch.idx}")
        self.surrogate.load_arch(arch)
        arch.acc = self.surrogate.predict_arch(arch.idx, self.regressor_type)
        #arch_obj.model = self.create_model(arch_obj)
        arch.acc_hist = []
        arch.loss_hist = []
        arch.loss = 0   # Final validation loss
        arch.cpu_hours = 0            # Training time in CPU-hours
        arch.num_params = 0 #calculate_model_params(model) # Total number of model parameters
        arch.flops = 0 #calculate_model_flops(model)
        arch.trained_epochs = 0
        arch.gen_creation = self.generation
        #if report == True and arch.archStatus == 'BEST':
        arch.flops, arch.num_params, arch.sizeMB = self.flops_params_gen_list_str(get_realMetrics = False, gen_list_str = str(arch.genotype.gen_list))
        self.log_out_of_bounds_arch(arch)
        self.reporter.save_arch_info(self, arch) if config_tecnas.REPORT_ONLY_BEST == False else None
        print2(Fore.GREEN + f'\nTraining {arch.idx} complete...\n')
    
    
    def real_training(self, arch, epochs = ConfigClass.EPOCHS, calculate_cm = False, report = True):
        start_time = time.time()
        with tf.device('/gpu:0'):
            tf.keras.backend.clear_session()
            arch.model = self.create_model(arch)
            #arch.model.summary()
            #arch_obj.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            arch.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=5e-4),  loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            history = arch.model.fit(self.train_generator, batch_size=Globals.BATCH_SIZE, epochs=epochs, validation_data=self.validation_generator, callbacks=[reduce_lr], verbose = 1)
            # Evaluate the model on test data
            print(f'Evaluating {arch.idx} on test data...')
            test_loss, test_acc = arch.model.evaluate(self.x_test, self.y_test, verbose=0)
            arch.top1 = test_acc
            arch.top5 = self.compute_topk(arch.model, self.x_test, self.y_test, k=5)
            print(f"Top-1 Accuracy: {arch.top1}")
            print(f"Top-5 Accuracy: {arch.top5}")
            print(f'\nTraining {arch.idx} complete')
         
        # Calculate and print CPU-Hours
        arch.acc_hist = history.history['val_accuracy']
        arch.loss_hist = history.history['val_loss']
        #arch.acc = history.history['val_accuracy'][-1]  # Max validation accuracy
        arch.acc = max(history.history['val_accuracy']) # Max validation accuracy
        arch.loss = history.history['val_loss'][-1]     # Final validation loss
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
            arch.flops, arch.num_params, arch.sizeMB = self.flops_params_gen_list_str(getflops = True, get_numparams = True, gen_list_str = str(arch.genotype.gen_list))
            print(Fore.YELLOW + f'FLOPs, number of parameters and size in MB for {arch.idx} calculated: {arch.flops}, {arch.num_params}, {arch.sizeMB}MB')
            
        if report == True:
            self.reporter.save_arch_info(self, arch)
        print(Fore.GREEN + f'\nTraining {arch.idx} complete...\n')
        end_time = time.time()
        elapsed_min = (end_time - start_time)/60
        print(Fore.LIGHTBLUE_EX + f"{elapsed_min:.1f} min.")
        return arch
    
    def train_model(self, arch = None, calculate_cm = False, epochs = ConfigClass.EPOCHS, print_status = True):
        def set_search_str():
            hhse_str = ""
            if config_tecnas.HHSE_RANDOM:
                hhse_str = "RANDOM"
            elif config_tecnas.HHSE_MARKOV:
                markov_mode = "COMBINED" if self.COMBINED_MARKOV else self.markov_type
                hhse_str = f"MARKOV {markov_mode}"
            elif config_tecnas.HHSE_GREEDY:
                hhse_str = f"GREEDY: {config_tecnas.HHSE_GREEDY_CRITERIA}"
            elif config_tecnas.HHSE_TEC:
                hhse_str = f"AOS: MAB UCB"
            else:
                hhse_str = ""
            return hhse_str
        ast_bar = self.ast_bar

        if self.generation > 0:
            if print_status == True:
                #print(Fore.LIGHTBLUE_EX + f'\n{ast_bar} EXECUTION {self.exec}/{ConfigClass.EXECUTIONS} GENERATION {self.generation}/{ConfigClass.GENERATIONS} {self.crossover_type} {self.mutation_type} {self.arch_count+1}/{ConfigClass.TOTAL_ARCH} architectures {ast_bar}')
                hhse_str = set_search_str()
                print1(Fore.LIGHTBLUE_EX + f'\n{ast_bar} EXECUTION {self.exec}/{ConfigClass.EXECUTIONS} GENERATION {self.generation}/{ConfigClass.GENERATIONS} {self.crossover_type}_{self.mutation_type}  | NSGA2 {self.NSGA2} | HHSE {self.HHSE} {hhse_str} {ast_bar}' + Fore.RESET)
            print1(Fore.YELLOW + f'{ConfigClass.DATASET} {ConfigClass.DATASET_PART*100}% |Repr Encoding: {self.representation_type}-{self.encoding_type}| |Surrogate: {self.SURROGATE}| |Train: {self.TRAIN}| |Simulate: {self.SIMULATE}| Surrogate folder: {self.experiment_folder}' + Fore.RESET)
            print1(Fore.YELLOW + f'Mutation prob: {self.MUT_PROB*100:.2f}% Hypermutation: {self.HYPER_MUTATION} Explore SCNG: {self.perc_explore_SCNG*100:.2f}% - Explore SCNF: {self.perc_explore_SCNF*100:.2f}% -- {"SCNG" if self.calculate_SCNG else ""} {"SCNF" if self.calculate_SCNF else ""} {"INT2" if self.INTEGER_ENCODING2 else "INT1"} {self.distance_type}' + Fore.RESET)
            print1(Fore.YELLOW + f'SCNG_THRESHOLD: {self.SCNG_THRESHOLD:.5f} | SCNF_THRESHOLD: {self.SCNF_THRESHOLD:.5f}' + Fore.RESET)
            print1('\n\n\n')
        
        self.arch_count += 1
        process = psutil.Process(os.getpid())
        mem_proceso = process.memory_info().rss  # bytes
        mem_total = psutil.virtual_memory().total  # bytes
        mem_perc = (mem_proceso / mem_total) * 100
        cpu_usage = psutil.cpu_percent(interval=None)
        print2(Fore.RED + f"RAM used: {mem_perc}%\n CPU used: {cpu_usage}%" + Fore.RESET)
        self.RAM = f'{mem_perc}'
        self.CPU = f'{cpu_usage}'
        print2(Fore.GREEN + f'\nTraining {arch.idx}...')
        print2(arch.genotype.gen_list)
        #if ConfigClass.REPRESENTATION_TYPE == 'B': #
        #    arch.genotype.gen_list = BlockRepresentation.reset_add_pools(arch.genotype.gen_list)
        arch.set_report_path(tecnasObj = self)
        print()
        if self.SIMULATE == True:
            self.simulate_training(arch, report = self.report_training)
        elif self.SURROGATE == True:
            self.training_with_surrogate(arch, report = self.report_training)
        elif self.TRAIN == True:
            self.real_training(arch, epochs = epochs, calculate_cm = calculate_cm, report = self.report_training)
        else:
            print(Fore.LIGHTRED_EX + f'ERROR: No training method selected' + Fore.RESET)
            return None
        return arch
    
    def set_arch_statuses(self, distance_type = 'None'):
        #Identify BEST, WORST AND MEDIAN. Save their info and reset their status. Also, calculate penalization for BEST.
        #Saving info about the WORST and MEDIAN ARCHITECTURES (BEFORE ELITISM) -------------------------- CHECKING THIS
        #Also store the worst arch and median
        print2('\nIdentifying BEST, MEDIAN and WORST architectures')
        self.pop[-1].archStatus = 'WORST'
        lower_median_idx = self.get_lower_median_architecture_idx(self.pop)
        self.pop[lower_median_idx].archStatus = 'MEDIAN'
        self.pop[0].archStatus = 'BEST'

        if self.generation > 1:
            #self.pop[0].dPB = hamming_distance(self.prev_best_integerEnc, self.pop[0].integer_encoding)/len(self.pop[0].integer_encoding)
            vec_prev_best = self.get_vector_for_distances(arch=self.prev_best_arch)
            vec_pop0 = self.get_vector_for_distances(arch=self.pop[0])
            distance = get_distance(vector=vec_prev_best,matrix=np.asarray(vec_pop0).reshape(1, -1), distance_type=distance_type)
            self.pop[0].dPB = distance[0] 
            #self.pop[0].dPB = get_distance(vectors_list = self.get_vectors_for_distances(arch1 = self.prev_best_arch, arch2 = self.pop[0]), distance_type = distance_type)
        else:
            print2('First generation, no previous best architecture')
            self.pop[0].dPB = 0
        self.prev_best_arch = copy.deepcopy(self.pop[0])
        #self.reporter.save_arch_info(self, self.pop[-1]) #Worst
        #self.reporter.save_arch_info(self, self.pop[lower_median_idx]) #Median
        self.pop[-1].archStatus = ''
        self.pop[lower_median_idx].archStatus = ''
        self.get_NFHT(self.pop[0])
        self.pop[0].flops, self.pop[0].num_params, self.pop[0].sizeMB = self.flops_params_gen_list_str(get_realMetrics = False, gen_list_str = str(self.pop[0].genotype.gen_list))
        #resetInfo is used for resetting some info about the best arch. saveRatios is for saving succ cross and succ mut.
        #print(f'set_arch_statuses function BEST: {self.pop[0].idx = }{self.pop[0].integer_encoding = }{self.pop[0].acc = }')
        #for arch in self.pop:
        #    print(f'{arch.idx = }{arch.integer_encoding = }{arch.acc = }{arch.archStatus = }')
        #print(f'set_arch_Statuses finish')
        #time.sleep(10)
        self.reporter.save_arch_info(self, self.pop[0]) #Best
        self.pop[0].archStatus = ''
        print2('\nIdentifying BEST, MEDIAN and WORST architectures complete')
                            
    
    def get_NFHT(self, arch_obj):
        #Normalized First Hitting Time.
        arch_obj.NFHT = arch_obj.gen_creation/ConfigClass.GENERATIONS


    def initialize_pop(self):
        print2('\nInitializing population')
        self.pop = []
        self.pool = []
        self.children = []
        print2('Generating and training initial population')
        for i in range(self.NPOP):
            random_arch = self.random_individual()
            random_arch = self.train_model(random_arch)
            self.pop.append(random_arch)
        print2('Generating and training initial population complete')

    
    def select_HHSE_method(self):
        #if config_tecnas.HHSE_Tree == True:
            #Select the operator by means of random forests.
        #    self.crossover_type, self.mutation_type = self.HHSE_Random() if self.generation == 1 else self.HHSE_Tree()
        if config_tecnas.HHSE_RANDOM == True:
            #Select the operator randomly
            self.crossover_type, self.mutation_type = self.HHSE_Random()
        elif config_tecnas.HHSE_GREEDY == True:
            #Calculates all combinations and selects the best one.
            self.crossover_type, self.mutation_type =  self.HHSE_GREEDY()
            self.set_arch_statuses(distance_type = self.distance_type) #This is done here because the greedy function applies all GOs and needs to know which one is best.
        elif config_tecnas.HHSE_MARKOV == True:
            self.current_state = self.HHSE_MARKOV()
            self.crossover_type, self.mutation_type = self.current_state.split('_')
        elif config_tecnas.HHSE_TEC == True:
            self.crossover_type, self.mutation_type = self.HHSE_TEC()

        else:
            print(Fore.RED + 'No HHSE method selected. Aborting' + Fore.RESET)
            return None
        #self.operator = f'{self.crossover_type}_{self.mutation_type}'
        self.operator = f'{self.crossover_type}_{self.mutation_type}'
        if not config_tecnas.HHSE_GREEDY: #This op count is done inside the greedy function.
            self.op_count[self.operator] = self.op_count.get(self.operator, 0) + 1
        
        print2(Fore.MAGENTA + f'Crossover: {self.crossover_type}, Mutation: {self.mutation_type}' + Fore.RESET) 

    def HHSE_Tree(self):
        tree = joblib.load(r"C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\trees\trees\operator_xgboost.pkl")
        le = joblib.load(r"C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\trees\trees\operator_encoder.pkl")
        statusObj = Status(tecnasObj = self, pop = self.pop)
        current_state = delete_columns_dict(statusObj.dict_archinfo, config_tecnas.remove_status_columns) #NOT WORKING WTF
        current_state = {k: v[0] if isinstance(v, list) else v for k, v in current_state.items()} #Make all values numeric (they are arrays, for the report.)
        X_state = pd.DataFrame([current_state])
        predicted_class = tree.predict(X_state)[0]
        self.operator = le.inverse_transform([predicted_class])[0]
        cross_type, mut_type = max(self.op_count, key = self.op_count.get).split('_') if self.operator == 'NONE' else self.operator.split('_')
        return cross_type, mut_type
        #self.crossover_type, self.mutation_type = ('TPC', 'MPAR') if self.operator == 'NONE' else self.operator.split('_')

    def HHSE_Random(self):
        return random.choice(self.crossover_types_list), random.choice(self.mutation_types_list)

    def HHSE_GREEDY(self):
        print('Applying all genetic operators')
        best_go, self.pop = self.apply_all_GOs()
        print('Application of all genetic operators complete.')
        return best_go.split('_')
    
    def HHSE_TEC(self):
        self.AOS_rwd_percentages = [0, 0, 0] #For calculating the reward in TECNAS AOS. [HV, CROSS, MUT]
        if self.generation == 1:
            cross, mut = 'NONE', 'NONE' #Next step is to apply rest of GOs to calculate UCB info.
            return cross, mut
        if config_tecnas.HHSE_TEC_IMPROVED:
            self.CROSS_GOs, self.MUT_GOs, self.HV_GOs = config_tecnas.set_HHSE_TEC_GOs(self.representation_type, self.encoding_type)
            progress = self.generation / self.total_gen
            if progress <= 0.40:
                cross, mut = self.TECNAS_AOS_obj.select_operator(candidates=self.MUT_GOs).split('_')
                self.AOS_rwd_percentages = [0.30, 0.10, 0.60] 
            elif progress <= 0.75:
                cross, mut = self.TECNAS_AOS_obj.select_operator(candidates=self.HV_GOs).split('_')
                self.AOS_rwd_percentages = [0.80, 0.10, 0.10]
            elif progress <= 1.0:
                cross, mut = self.TECNAS_AOS_obj.select_operator(candidates=self.CROSS_GOs).split('_')
                self.AOS_rwd_percentages = [0.30, 0.60, 0.10]
        else:
            cross, mut = self.TECNAS_AOS_obj.select_operator().split('_')
        return cross, mut

        def considerlater(self):
            #Choose the GO, according to the probability.
            #CROSS_DICT = {'SPC_MPOLY': 0.45, 'SPC_MNUF': 0.35, 'SPC_NONE': 0.2}
            self.CROSS_DICT, self.MUT_DICT, self.HV_DICT, self.GD_DICT = config_tecnas.set_HHSE_TEC_GOs(self.representation_type, self.encoding_type)
            progress = self.generation / self.total_gen
            
            if progress <= self.limits[0]:
                next_go = random.choices(list(self.MUT_DICT.keys()), weights=self.MUT_DICT.values(),  k=1)[0]
            elif progress <= self.limits[1]:
                next_go = random.choices(list(self.CROSS_DICT.keys()), weights=self.CROSS_DICT.values(),  k=1)[0]
            elif progress <= self.limits[2]:
                next_go = random.choices(list(self.HV_DICT.keys()), weights=self.HV_DICT.values(),  k=1)[0]
            elif progress <= self.limits[3]:
                next_go = random.choices(list(self.GD_DICT.keys()), weights=self.GD_DICT.values(),  k=1)[0]
            else:
                next_go = 'ERROR'
                print(Fore.RED + 'Error in HHSE_TEC: progress exceeds limits' + Fore.RESET)
            return next_go.split('_')

    def NSGA2_window(self):
        progress = self.generation / self.total_gen
        if self.generation == 1:
            self.original_cross_type = self.crossover_type
            self.original_mut_type = self.mutation_type
        
        if progress <= config_tecnas.NSGA2_WINDOW_SIZE_PERC:           
            self.crossover_type, self.mutation_type = random.choice(config_tecnas.crossoverList_hhse), random.choice(config_tecnas.mutationList_hhse)
        else:
            self.crossover_type, self.mutation_type = self.original_cross_type, self.original_mut_type
        
    def mainLoop(self):
        if self.HHSE == True:
            if config_tecnas.HHSE_RANDOM or config_tecnas.HHSE_GREEDY:
                self.ENAS()
            elif config_tecnas.HHSE_TEC:
                self.ENAS()
            else: #MARKOV
                for markov_type in config_tecnas.HHSE_MARKOV_TYPES:
                    self.COMBINED_MARKOV = False
                    if markov_type == 'COMBINED':
                        self.COMBINED_MARKOV = True
                    self.markov_type = markov_type
                    self.ENAS()
        else: #No HHSE selected
            for crossover_type in ConfigClass.crossover_types_list:
                self.crossover_type = crossover_type
                for mutation_type in ConfigClass.mutation_types_list:
                    self.mutation_type = mutation_type
                    print(crossover_type, mutation_type)
                    self.ENAS()
        
            
        print2('Ranking architectures')
        rank_archs(self.children[0].path_filereport)
        print2('Ranking finished\n')
        print2('Filtering results by best architectures')
        #filter_csv(self.children[0].path_folder, generation = 'LAST', arch_status = 'BEST', rank = 'ALL', nparts = 1) 
        print2('Filtering complete')
        print2('Filtering by rank 1 architectures')
        #filter_csv(self.children[0].path_folder, generation = 'LAST', arch_status = 'BEST', rank = 'HIGHEST', nparts = 1)
        print2('Filtering by rank 1 complete')

    def ENAS(self):
        self.set_search_name()
        if self.crossover_type == 'NONE' and self.mutation_type == 'NONE':
            self.search_strategy = 'RANDOM'
        else:
            self.search_strategy = 'GA'
        seed_idx = 0
        self.prev_best_arch = []
        if not self.validate_encoding_genetic_operators(): return None
        for e in range(1,ConfigClass.EXECUTIONS+1):
            self.SCNF_history = []
            self.SCNG_history = []
            self.local_seed = self.SEED_LIST[seed_idx]
            random.seed(self.local_seed)
            np.random.seed(self.local_seed)
            #tf.random.set_seed(self.local_seed)
            #self.MUT_PROB = ConfigClass.MUT_PROB
            self.exec = e
            self.trained_archs = {}
            self.generation = 1
            self.initialize_pop()
            self.num_unchangedPB = 0
            self.op_count = {}
            self.GOs_history = []
            self.GOs_HV = {k: [0, 0, 0] for k in self.GOs_HV} #HV, PREV_HV and dHV.
            self.TECNAS_AOS_obj = TECNAS_AOS(tecnasObj = self)
            self.encoding_vectors_history = set()
            self.objectives_history = set()
            
            for g in range(1,ConfigClass.GENERATIONS+1):
                self.pool = []
                self.generation = g
                self.NSGA2_window() if config_tecnas.NSGA2_WINDOW  else None
                if self.HHSE == True: #Choose crossover and mutation type automatically
                    print2(Fore.MAGENTA + 'Hyper-heuristic selection of genetic operators' + Fore.RESET)
                    self.select_HHSE_method()
                if not config_tecnas.HHSE_GREEDY: #This means the new gneration hasn't been generated yet. If this is true, GREEDY has already calculated the best self.pop
                    ########### HYPER MUTATION ##########################################
                    if self.HYPER_MUTATION:
                        if self.perc_explore_SCNG <= self.HYPER_MUT_THRESHOLD: #Exploration starts to weaken and exploiting started. 
                            self.MUT_PROB = self.HYPER_MUT_PROB
                            print(Fore.RED + f'Hyper mutation applied {self.perc_explore_SCNG = }. New mutation probability: {self.MUT_PROB}' + Fore.RESET)
                        else:
                            self.MUT_PROB = self.ORIGINAL_MUT_PROB

                    ################# BALANCING ##########################################
                    #self.search_name = 'TEC_BALANCE3'
                    #cross_list = ['SPC', 'TPC', 'UC', 'SBX']
                    #mut_list = ['MPOLY', 'MNUF']
                    #if self.perc_exploit_SCNG > 0.5: #It means exploration <= 0.5  Stop exploiting, start exploration
                    #    self.crossover_type = random.choice(cross_list)
                    #    self.mutation_type = 'MNUF'
                    #else: #Means exploitation <= 0.5 and exploration is > 0.5 #Continue exploiting
                    #    self.crossover_type = random.choice(cross_list)
                    #    self.mutation_type = 'MPOLY'
                    ################# BALANCING ##########################################
                    self.genetic_operators()
                    self.report_training = True
                    self.pool = self.pop + self.children

                    self.save_encoding_vectors_history(self.pool) if self.calculate_SCNG else None
                    self.save_objectives_history(self.pool) if self.calculate_SCNF else None
                    self.sorted_pool = sorted(self.pool, key=lambda obj: obj.acc, reverse = True) #Best archs first
                    self.pop = self.selection(pop = copy.deepcopy(self.pop)) #Here, WORST AND MEDIAN ARCHS ARE IDENTIFIED
                    self.GD = self.calculate_GD(archs_list = self.pop, distance_type = self.distance_type)
                    
                    self.calculate_SCN(SCN_G = self.calculate_SCNG, SCN_F = self.calculate_SCNF) if self.calculate_SCNG or self.calculate_SCNF else None
                    
                    self.reporter.save_generation_status(Status(tecnasObj = self, crosstype = self.crossover_type, muttype = self.mutation_type, pop = self.pop, JUST_PARETO = True)) #Save the info from the pareto front.
                    self.set_arch_statuses(distance_type = self.distance_type)

                if config_tecnas.HHSE_TEC:
                    if self.generation > 1:
                        self.TECNAS_AOS_obj.total_selections += 1 if config_tecnas.HHSE_TEC else 0
                        self.TECNAS_AOS_obj.update_arm(GO_str = f'{self.crossover_type}_{self.mutation_type}', reward = self.TECNAS_AOS_obj.calculate_reward(*self.AOS_rwd_percentages)) if config_tecnas.HHSE_TEC else None
                       
            self.report_op_counts(results_folder = self.children[0].path_folder) if config_tecnas.HHSE_GREEDY_FIXED else None
            self.generate_total_counts(os.path.join(self.children[0].path_folder, f"counts_{config_tecnas.HHSE_GREEDY_CRITERIA}")) if config_tecnas.HHSE_GREEDY_FIXED else None
            self.reporter.flush(self.pop[0].path_filereport)
            seed_idx += 1 if seed_idx < len(self.SEED_LIST)-1 else 0
            self.local_seed = self.SEED_LIST[seed_idx]
            self.save_histories()
            self.TECNAS_AOS_obj.report_arms_csv(results_folder = self.children[0].path_folder) if config_tecnas.HHSE_TEC else None
            self.SCN_stats_histogram()

                       
            

    def __init__(self, HHSE = False, NSGA2 = False, report_training = True, representation_type = ConfigClass.REPRESENTATION_TYPE, encoding_type = ConfigClass.ENCODING_TYPE, experiment_folder = '', regressor_type = -1):

        def set_MUT_PROB(INTEGER_ENCODING2):
            #Create a random arch and get its vector length. God, this is so stupid.
            arch = self.random_individual()
            if self.encoding_type == 'BIN':
                vec_bin = arch.binary_encoding2_flattened if INTEGER_ENCODING2 else arch.binary_encoding 
                prob_mut = 1/sum(len(v) for v in vec_bin)
            elif self.encoding_type == 'INT':
                vec_int = arch.integer_encoding2_flattened if INTEGER_ENCODING2 else arch.integer_encoding
                prob_mut = 1/len(vec_int)
            elif self.encoding_type == 'REAL':
                vreal = arch.real_encoding2_flattened if INTEGER_ENCODING2 else arch.real_encoding
                prob_mut = 1/len(vreal)
            return prob_mut

        def set_distance(encoding_type):
            if encoding_type == 'BIN':
                return 'hamming'
            elif encoding_type == 'INT':
                return 'manhattan' if config_tecnas.INTEGER_ENCODING2 else 'family'
            elif encoding_type == 'REAL':
                return 'manhattan' if config_tecnas.INTEGER_ENCODING2 else 'family'
            else:
                print(Fore.LIGHTRED_EX + f'ERROR: {encoding_type} Encoding type not recognized' + Fore.RESET)
                return None
        #SCN THRESHOLD
        #BIN INT1:  SCNF 0.0080 SCNG 0.030
        #INT INT1:  SCNF 0.0063 SCNG 0.050
        #REAL INT1: SCNF 0.0076 SCNG 0.070
        #BIN INT2:  SCNF 0.0083 SCNG 0.030
        #INT INT2:  SCNF 0.0080 SCNG 0.035
        #REAL INT2: SCNF 0.0080 SCNG 0.035
        self.SCNF_THRESHOLDS = {'BIN_INT1' : 0.0080,  'BIN_INT2': 0.0083,'INT_INT1': 0.0063, 'INT_INT2': 0.0080, 'REAL_INT1': 0.0076, 'REAL_INT2': 0.0080}
        self.SCNG_THRESHOLDS = {'BIN_INT1' : 0.030,  'BIN_INT2': 0.030,'INT_INT1': 0.050, 'INT_INT2': 0.035, 'REAL_INT1': 0.070, 'REAL_INT2': 0.0350}
        #{'BIN_INT1' : 0.26,  'BIN_INT2': 0.011,'INT_INT1': 0.26, 'INT_INT2': 0.011,
                               # 'REAL_INT1': 0.26, 'REAL_INT2': 0.011}

        
        #config_tecnas CONSTANTS################################################
        self.REPR_ENC_STR = f'{representation_type}{encoding_type}'
        self.SEED_LIST = config_tecnas.SEED_LIST
        self.calculate_SCNG = config_tecnas.calculate_SCNG
        self.calculate_SCNF = config_tecnas.calculate_SCNF
        self.NSGA2 = config_tecnas.NSGA2
        self.SCNG_THRESHOLD = self.SCNG_THRESHOLDS[f'{encoding_type}_INT{2 if config_tecnas.INTEGER_ENCODING2 else 1}']
        self.SCNF_THRESHOLD = self.SCNF_THRESHOLDS[f'{encoding_type}_INT{2 if config_tecnas.INTEGER_ENCODING2 else 1}']
        self.HHSE_GREEDY_CRITERIA = config_tecnas.HHSE_GREEDY_CRITERIA
        self.HYPER_MUTATION = config_tecnas.HYPER_MUTATION
        self.HYPER_MUT_PROB = config_tecnas.HYPER_MUT_PROB
        self.HYPER_MUT_THRESHOLD = config_tecnas.HYPER_MUT_THRESHOLD
        #self.distance_type = 'hamming' if encoding_type == 'BIN' else self.distance_type

        self.distance_type = set_distance(encoding_type)
        ############################################################################

        self.INTEGER_ENCODING2 = config_tecnas.INTEGER_ENCODING2
        self.search_name = '' #Indicates type of search strategy and encoding. For example, LREAL_SPC_MPAR_NH, BREAL_MARKOV_MUT_NH (N means NSGA2, H means HHSE).
        self.report_training = report_training
        self.HHSE = HHSE #Hyper-heuristic selection of genetic operators
        self.NSGA2 = NSGA2
        self.ast_bar = 50*'+'
        self.representation_type = representation_type #Layer (L), Block (B)
        self.encoding_type = encoding_type
        
        self.NPOP = ConfigClass.MAIN_NPOP
        self.MUT_PROB = set_MUT_PROB(self.INTEGER_ENCODING2)
        self.ORIGINAL_MUT_PROB = self.MUT_PROB #This is because maybe MUT_PROB may change
        self.crossover_types_list = ConfigClass.crossover_types_list if HHSE == False else config_tecnas.crossoverList_hhse
        self.mutation_types_list = ConfigClass.mutation_types_list if HHSE == False else config_tecnas.mutationList_hhse
        self.crossover_type = None
        self.mutation_type = None
        self.original_cross_type, self.original_mut_type = None, None #For NSGA2 window
        self.search_strategy = None
        self.reporter = ReportENAS()
        self.generation = 0
        self.exec = 0
        self.arch_count = 0
        self.succ_mut = 0
        self.succ_cross = 0
        self.total_mut = 0
        self.total_cross = 0
        self.GD = 0 #Generational Distance
        self.prev_best_arch = None
        self.pop = []
        self.children = []
        self.pool = [] #Parents + children
        self.trained_parents = []
        self.flops_archs = {} #Dictionary with architectures whose FLOPs and NumParams were already calculated
        self.accuracy_histories = [] #To plot all accuracies through epochs
        self.loss_histories = [] #To plot all accuracies through epochs
        self.encoding_vectors_history = set() #For calculating SCN_G. I use a set to avoid duplicates.
        self.objectives_history = set() #For calculating SCN_F.
        self.NEXPLOIT_SCNG = 0
        self.NEXPLORE_SCNG = 0
        self.perc_explore_SCNG = 1 
        self.perc_exploit_SCNG = 0
        self.NEXPLORE_SCNF = 0
        self.NEXPLOIT_SCNF = 0
        self.perc_explore_SCNF = 1
        self.perc_exploit_SCNF = 0
        self.SCNF_history = []
        self.SCNG_history = []
        
        self.NUM_DENSE = ConfigClass.DATASET_NUMDENSE_DICT[ConfigClass.DATASET]
        self.MAXINT = Globals.INDEXES_POOLS[-1] if self.representation_type == 'L' else len(BlockRepresentation.all_blocks) - 1
        
        #============================================================= MAX MIN FLOPS AND PARAMS FOR HV NORMALIZATION ==============================================================
        self.BLOCKS_MINFLOPS  = 1.24e8
        self.BLOCKS_MAXFLOPS  = 1.6e10   #avg: 3e9
        self.BLOCKS_MINPARAMS = 1.6e5
        self.BLOCKS_MAXPARAMS = 2.51e7   #avg: 6e6

        self.LAYERS_MINFLOPS  = 1e6
        self.LAYERS_MAXFLOPS  = 1.71e10
        self.LAYERS_MINPARAMS = 1e3
        self.LAYERS_MAXPARAMS = 1.42e8

        self.MINFLOPS =  self.BLOCKS_MINFLOPS  if self.representation_type == 'B' else self.LAYERS_MINFLOPS
        self.MINPARAMS = self.BLOCKS_MINPARAMS if self.representation_type == 'B' else self.LAYERS_MINPARAMS
        self.MAXFLOPS =  self.BLOCKS_MAXFLOPS  if self.representation_type == 'B' else self.LAYERS_MAXFLOPS
        self.MAXPARAMS = self.BLOCKS_MAXPARAMS if self.representation_type == 'B' else self.LAYERS_MAXPARAMS

        #For HHSE ========================================= HHSE ========================================================================
        self.TECNAS_AOS_obj = None #For MAB and UCB.
        self.AOS_rwd_percentages = [0,0,0] #For calculating the reward in TECNAS AOS. [accuracy, GD, HV]
        #For GREEDY, only chose the identified GOs for each stage.
        self.GOs_names = [f"{cross}_{mut}" for cross, mut in product(self.crossover_types_list, self.mutation_types_list)] 
    
        if config_tecnas.IGNORE_RANDOM and 'NONE_NONE' in self.GOs_names:
            self.GOs_names.remove('NONE_NONE')
        #Save here all population generated through all combinations of crossover and mutation
        #For example, = {'SPC_MPAR': [arch1, arch2, ...], 'SPC_MSWAP': [arch1, arch2, ...], ...}
        self.GOs_names = sum(config_tecnas.set_HHSE_TEC_GOs(self.representation_type, self.encoding_type), []) if config_tecnas.HHSE_TEC else self.GOs_names
        self.GOs_population_dict = {}
        self.selected_by = ' ' #Depicts why the GO was selected as the best one in HHSE GREEDY
        self.GOs_history = [] #['SPC_MPAR', 'TPC_MSWAP', 'UC_MPAR', 'TPC_MSWAP',...} GOs order of application
        self.GOs_HV = {k: [0, 0, 0] for k in self.GOs_names} #HV, PREV_HV and dHV.
        
        self.perc_intervals = config_tecnas.perc_intervals  #This is for setting the time intervals for COMBINED MARKOV.
        self.total_gen = config_tecnas.GENERATIONS
        self.limits = []
        self.cumulative = 0
        for p in self.perc_intervals:
            self.cumulative += p / 100
            self.limits.append(self.cumulative)

               
        #================================================= HHSE ========================================================================

        self.HV = 0 #Hypervolume of the pareto front in the current generation
        self.riesz_energy = 0 #Riesz energy of the pareto front in the current generation
        self.prev_HV = 0 #previous HV for calculating delta HV
        self.dHV = 0
        self.report_columns = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs", "Acc_mean", "Loss_mean", "CPU_hrs_mean", "Num_params_mean", "FLOPs_mean", "Acc_std", "Loss_std", "CPU_hrs_std", "Num_params_std","FLOPs_std"]
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
        self.getflops = False
        self.get_numparams = False
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
        self.surrogate = Surrogate_ENAS(self.regressor_type, self.regressor_folder) if self.regressor_type >=0 else None
        self.SURROGATE = ConfigClass.SURROGATE
        self.TRAIN = ConfigClass.TRAIN
        self.SIMULATE = ConfigClass.SIMULATE
        self.REPORT_ARCH = ConfigClass.REPORT_ARCH

        if self.TRAIN == True:
            self.get_normalize_dataset()

        self.current_dir =  os.getcwd()  #'/content/drive/MyDrive/DCC/Research/First_article'
        self.dir_archs = 'architectures'
        self.path_results = os.path.join(path, self.dir_archs)
        
    
'''
gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[128,'relu']}, {'LAST_DENSE':[10,'softmax']}]
archL = LayerRepresentation(genotypeObj = Genotype(rep_type='L', encoding_type='INT', gen_list=gen_list))
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
0:  ({'CONV': [32, 3]}, {'CONV': [32, 3]})
1:  ({'CONV': [32, 3]}, {'CONV': [32, 5]})
2:  ({'CONV': [32, 3]}, {'CONV': [64, 3]})
3:  ({'CONV': [32, 3]}, {'CONV': [64, 5]})
4:  ({'CONV': [32, 3]}, {'CONV': [128, 3]})
5:  ({'CONV': [32, 3]}, {'CONV': [128, 5]})
6:  ({'CONV': [32, 3]}, {'CONV': [256, 3]})
7:  ({'CONV': [32, 3]}, {'CONV': [256, 5]})
8:  ({'CONV': [32, 5]}, {'CONV': [32, 3]})
9:  ({'CONV': [32, 5]}, {'CONV': [32, 5]})
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
