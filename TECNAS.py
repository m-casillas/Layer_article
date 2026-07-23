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

    def calculate_SCN(self, arch_integer_encoding):
        #Calculate Similariy of Closest Neighbbor of current individual with all generatios.
        min_dist = float('inf')
        for integer_encoding in self.integer_encoding_history:
            dist = hamming_distance(arch_integer_encoding, integer_encoding)
            if dist < min_dist:
                min_dist = dist
                if dist == 1: #No need to continue checking.
                    break
        return min_dist


    def calculate_GD(self, archs_list):
        #Calculate the Genotype Distance (GD) of a list of architectures.
        #GD is the mean Hamming distance of all architectures in archs_list.
        from itertools import combinations
        add = 0
        N = 0
        print1(Fore.MAGENTA + f"Calculating GD for {len(archs_list)} architectures...")
        for a, b in combinations(archs_list, 2):
            add += hamming_distance(a.integer_encoding, b.integer_encoding)
            N += 1
        print1(Fore.MAGENTA + f"GD calculation complete. Total pairs: {N}")
        GD = add/N
        GD = GD/BLOCKS_CONSTANTS.SIZE_BLOCKGEN if ConfigClass.REPRESENTATION_TYPE == 'B' else GD/LAYERS_CONSTANTS.SIZE_LAYERGEN
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
        elif config_tecnas.HHSE_MARKOV:
            self.search_name += f'MARKOV_{self.markov_type}'
        elif config_tecnas.HHSE_RANDOM:
            self.search_name += f'HRANDOM'
        elif config_tecnas.HHSE_TEC:
            self.search_name += f'HHSE_AOS'
        elif config_tecnas.NSGA2_NORMAL:
            nsga2_window_str = '_W' if config_tecnas.NSGA2_WINDOW else ''
            nsga2_window_size_str = f"{config_tecnas.NSGA2_WINDOW_SIZE_PERC*100:.0f}_" if config_tecnas.NSGA2_WINDOW else ''
            self.search_name += f'{self.crossover_type}_{self.mutation_type}{nsga2_window_str}{nsga2_window_size_str}'

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
        arch_obj = LayerRepresentation(genotypeObj=Genotype(gen_list=gen_list)) if ConfigClass.REPRESENTATION_TYPE == 'L' else BlockRepresentation(genotypeObj=Genotype(gen_list=gen_list))
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

# ================================================================================= MARKOV===================================================================

    def markov_aggregate_matrix(self):
        import os
        import csv

        matrix_folder = os.path.join(self.pop[0].path_folder, "matrix")

        # Find all count matrix files
        files = [f for f in os.listdir(matrix_folder) if "count_matrix" in f and f.endswith(".csv")]

        if not files:
            raise ValueError("No count_matrix CSV files found in the folder.")

        aggregated_counts = None
        states = None

        # -------- Aggregate matrices --------
        for file in files:

            file_path = os.path.join(matrix_folder, file)

            with open(file_path, mode="r", newline="", encoding="utf-8") as f:

                reader = csv.reader(f)
                header = next(reader)
                current_states = header[1:]

                if aggregated_counts is None:

                    states = current_states
                    n = len(states)
                    aggregated_counts = [[0]*n for _ in range(n)]

                if current_states != states:
                    raise ValueError(f"State mismatch in file {file}")

                for i, row in enumerate(reader):

                    counts_row = list(map(int, row[1:]))

                    for j in range(len(counts_row)):
                        aggregated_counts[i][j] += counts_row[j]

        # -------- Remove zero rows --------
        row_sums = [sum(row) for row in aggregated_counts]

        valid_indices = [i for i, s in enumerate(row_sums) if s > 0]

        if not valid_indices:
            return [], [], []

        states = [states[i] for i in valid_indices]

        filtered_counts = []

        for i in valid_indices:
            filtered_row = [aggregated_counts[i][j] for j in valid_indices]
            filtered_counts.append(filtered_row)

        aggregated_counts = filtered_counts
        n = len(states)

        # -------- Build stochastic matrix --------
        P = []

        for row in aggregated_counts:

            total = sum(row)

            if total == 0:
                P.append([0]*n)
            else:
                P.append([value/total for value in row])

        # -------- Save COUNT matrix --------

        counts_file = os.path.join(
            matrix_folder,
            f"counts_{config_tecnas.REPRESENTATION_TYPE}{config_tecnas.ENCODING_TYPE}_"
            f"{config_tecnas.HHSE_GREEDY_CRITERIA}_Seed{config_tecnas.INITIAL_SEED}.csv"
        )

        with open(counts_file, mode="w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow(["counts"] + states)

            for i, row in enumerate(aggregated_counts):
                writer.writerow([states[i]] + row)

        # -------- Save stochastic matrix --------

        prob_file = os.path.join(
            matrix_folder,
            f"matrix_{config_tecnas.REPRESENTATION_TYPE}{config_tecnas.ENCODING_TYPE}_"
            f"{config_tecnas.HHSE_GREEDY_CRITERIA}_Seed{config_tecnas.INITIAL_SEED}.csv"
        )

        with open(prob_file, mode="w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([f"{config_tecnas.HHSE_GREEDY_CRITERIA}"] + states)

            for i, row in enumerate(P):
                writer.writerow([states[i]] + [f"{p:.6f}" for p in row])

        return states, aggregated_counts, P

    
    def markov_barplot_mean_stationary_probabilities(self):
        if config_tecnas.HHSE_GREEDY == False:
            return
        #Read the mean stationary probabilities from the CSV file and create a bar plot.
        report_folder = os.path.join(self.pop[0].path_folder, "matrix")
        mean_prob_file = os.path.join(report_folder, f"mean_stationary_probabilities.csv")
        if not os.path.exists(mean_prob_file):
            print(Fore.LIGHTRED_EX + "Mean stationary probabilities file not found.")
            return

        go_names = []
        mean_probs = []
        with open(mean_prob_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                go_names.append(row["GO_Name"])
                mean_probs.append(float(row["Mean_Stationary_Probability"]))

        plt.figure(figsize=(10, 6))
        plt.bar(go_names, mean_probs, color='skyblue')
        plt.xlabel('Genetic Operators')
        plt.ylabel('Mean Stationary Probability')
        plt.title(f'Mean Stationary Probabilities of GOs - {config_tecnas.HHSE_GREEDY_CRITERIA} Selection')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_file = os.path.join(report_folder, f"mean_stationary_probabilities.png")
        plt.savefig(plot_file)
        plt.close()
        print(Fore.GREEN + f'Mean stationary probabilities bar plot saved to {plot_file}')

    def markov_mean_stationary_probabilities(self):
        if config_tecnas.HHSE_GREEDY == False:
            return
        #Calculate the mean stationary probabilities for each GO across all generations and save them to a CSV file.
        report_folder = os.path.join(self.pop[0].path_folder, "matrix")
        os.makedirs(report_folder, exist_ok=True)
        # Read all stationary distribution files in the matrix folder
        stationary_files = [f for f in os.listdir(report_folder) if f.startswith("stationary_distribution_")]
        if not stationary_files:
            print(Fore.LIGHTRED_EX + "No stationary distribution files found.")
            return

        # Initialize a dictionary to store mean stationary probabilities for each GO
        mean_stationary_probs = {}
        # Initialize a list to store all probabilities for each GO
        all_probs = {}

        for filename in stationary_files:
            file_path = os.path.join(report_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    go_name = row["State"]
                    prob = float(row["Stationary_Probability"])
                    if go_name not in mean_stationary_probs:
                        mean_stationary_probs[go_name] = 0
                        all_probs[go_name] = []
                    mean_stationary_probs[go_name] += prob
                    all_probs[go_name].append(prob)

        # Calculate the mean of stationary probabilities for each GO
        num_files = len(stationary_files)
        for go_name in mean_stationary_probs:
            mean_stationary_probs[go_name] /= num_files

        # Write the mean stationary probabilities to a new CSV file
        output_file = os.path.join(report_folder, f"mean_stationary_probabilities.csv")
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["GO_Name", "Mean_Stationary_Probability"])
            for go_name, mean_prob in mean_stationary_probs.items():
                writer.writerow([go_name, f"{mean_prob:.6f}"])

    def markov_matrix(self, transitions=None, iterations=1000, tolerance=1e-10):
        if config_tecnas.HHSE_GREEDY == False:
            return

        matrix_folder = os.path.join(self.pop[0].path_folder, "matrix")
        ensure_folder_exists(matrix_folder)

        # --- include ALL possible states ---
        # STATES is assumed to be a global or imported list
        all_states = sorted(set(self.GOs_names))
        observed_states = set(transitions)

        # Index mapping for each state (ALL states included)
        index = {state: i for i, state in enumerate(all_states)}

        n = len(all_states)

        # Count matrix initialization
        counts = [[0] * n for _ in range(n)]

        # Count transitions ONLY for observed ones
        for i in range(len(transitions) - 1):
            current_state = transitions[i]
            next_state = transitions[i + 1]
            if current_state in index and next_state in index:
                counts[index[current_state]][index[next_state]] += 1

        # Transition probability matrix
        P = []
        for row in counts:
            total = sum(row)
            if total == 0:
                P.append([0] * n)
            else:
                P.append([value / total for value in row])

        # --- stationary distribution ONLY over observed states ---
        active_indices = [index[s] for s in all_states if s in observed_states]

        # initialize pi only on active states
        m = len(active_indices)
        pi = [0.0] * n
        if m > 0:
            for i in active_indices:
                pi[i] = 1.0 / m

            # power method restricted to active states
            for _ in range(iterations):
                new_pi = [0.0] * n
                for j in active_indices:
                    for i in active_indices:
                        new_pi[j] += pi[i] * P[i][j]

                # convergence check only on active states
                if max(abs(new_pi[i] - pi[i]) for i in active_indices) < tolerance:
                    pi = new_pi
                    break
                pi = new_pi

        # states not observed keep probability 0 automatically

        # Sort states by stationary probability (descending)
        ranking = sorted(zip(all_states, pi), key=lambda x: x[1], reverse=True)

        # File paths
        matrix_output_file = os.path.join(matrix_folder, f"markov_matrix_{self.exec}.csv")
        stationary_output_file = os.path.join(matrix_folder, f"stationary_distribution_{self.exec}.csv")
        counts_output_file = os.path.join(matrix_folder, f"count_matrix_{self.exec}.csv")

        # Write transition probability matrix to CSV (optional, still available)
        # with open(matrix_output_file, mode="w", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["State"] + all_states)
        #     for i, row in enumerate(P):
        #         writer.writerow([all_states[i]] + [f"{p:.6f}" for p in row])

        # Write stationary distribution to CSV
        with open(stationary_output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["State", "Stationary_Probability"])
            for state, prob in ranking:
                writer.writerow([state, f"{prob:.6f}"])

        # Write count matrix to CSV
        with open(counts_output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["State"] + all_states)
            for i, row in enumerate(counts):
                writer.writerow([all_states[i]] + row)

        return all_states, P, pi, ranking

    def selection(self, pop = None):
        print2('\nSelecting best parents and children for the new population')
        #Based on the SORT_ARCHS variable, select the best architectures from the sorted pool.
        if config_tecnas.NSGA2 == False:
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
            print('Selecting architectures with NSGA-II')
            self.nsga2_obj = TECNAS_NSGAII(pop = self.pool, pop_size = self.NPOP, tecnasObj = self)
            pop, self.HV = self.nsga2_obj.select_and_HV(self.pool)
            #self.HV = random.randint(1,30) #REMOVE THIS, IT IS JUST TO TEST THE GREEDY COMBINED SELECTION
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
            GOs_CROSS, GOs_MUT, GOs_HV, GOs_GD = config_tecnas.set_GREEDY_GOs_list(config_tecnas.REPRESENTATION_TYPE, config_tecnas.ENCODING_TYPE)
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
            GOs_CROSS, GOs_MUT, GOs_HV, GOs_GD = config_tecnas.set_GREEDY_FIXED_GOs_list(config_tecnas.REPRESENTATION_TYPE, config_tecnas.ENCODING_TYPE)
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
            self.GD = self.calculate_GD(self.GOs_population_dict[cross_mut_str]) if config_tecnas.HHSE_GREEDY_FIXED else 0
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
                child = self.make_child(child, arch, arch)
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
        child.dP1 = hamming_distance(child.integer_encoding, parent1.integer_encoding)/len(child.integer_encoding)
        child.dP2 = hamming_distance(child.integer_encoding, parent2.integer_encoding)/len(child.integer_encoding)
        child.dBM = 0

        child.P1acc = parent1.acc
        child.P2acc = parent2.acc
        child.P1flops = parent1.flops
        child.P2flops = parent2.flops
        child.P1params = parent1.num_params
        child.P2params = parent2.num_params
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
        archM.dBM = hamming_distance(archM.integer_encoding, archOriginal.integer_encoding)/len(archM.integer_encoding)
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

    
    
    def change_block(self, child, mutation_type):
        mutator_obj = Mutator()
        arch = BlockRepresentation(genotypeObj = None)
        arch.genotype = copy.deepcopy(child.genotype)
        mutable_block_indexes = BLOCKS_CONSTANTS.MUTABLE_BCHANGEPARAM_INDEXES
        
        if mutation_type == 'MBFLIP': #This is done separately because it works differently than the other mutations.
            arch.binary_encoding = mutator_obj.mutate_bitflip(child.binary_encoding, self.MAXINT)
            arch.binary_encoding_to_integer_encoding()
            arch.genotype.gen_list = arch.integer_vector_to_genList(arch.integer_encoding)
            arch.set_genotype(arch.genotype)
            return arch
        if mutation_type == 'MPOLY': #This one too, works differently.
            mpoly_indexes = range(len(BlockRepresentation().all_blocks)) #0 1 2 3... 63
            arch.real_encoding = mutator_obj.polynomial_mutation(arch_obj = child, t = self.generation, ylow = min(mpoly_indexes), yhigh = max(mpoly_indexes))
            arch.real_encoding_to_integer_encoding()
            arch.genotype.gen_list = arch.integer_vector_to_genList(arch.integer_encoding)
            arch.set_genotype(arch.genotype)
            return arch
        
        if mutation_type == 'MNUF': #This one too, works differently.
            mnounf_indexes = range(len(BlockRepresentation().all_blocks)) #0 1 2 3... 63
            arch.real_encoding = mutator_obj.nouniform_mutation(arch_obj = child, t = self.generation, ylow = min(mnounf_indexes), yhigh = max(mnounf_indexes))
            arch.real_encoding_to_integer_encoding()
            arch.genotype.gen_list = arch.integer_vector_to_genList(arch.integer_encoding)
            arch.set_genotype(arch.genotype)
            return arch

        if mutation_type == 'MPAR':
            mutation_function = mutator_obj.mutate_block_parameters
        elif mutation_type == 'MSWAP':
            mutation_function = mutator_obj.mutate_block_swap
       
        else:
            print(f'ERROR: {mutation_type} Mutation type not recognized for {config_tecnas.REPRESENTATION_TYPE}{config_tecnas.ENCODING_TYPE}')
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
                
        if child.encoding == 'INT':
            if mutation_type == 'MPAR':
                layers_indexes = LAYERS_CONSTANTS.MUTABLE_LCHANGEPARAM_INDEXES #Only this indexes may change their parameters
                mutation_function = mutator_obj.mutate_layer_parameters
            elif mutation_type == 'MSWAP':
                layers_indexes = LAYERS_CONSTANTS.MUTABLE_LSWAP_INDEXES #Only this indexes may change type
                mutation_function = mutator_obj.mutate_layer_swap
            else:
                print(Fore.RED + f'ERROR: {mutation_type} Mutation type not recognized for {config_tecnas.REPRESENTATION_TYPE}{config_tecnas.ENCODING_TYPE}' + Fore.RESET)
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
                print(Fore.RED + f'ERROR: {mutation_type} Mutation type not recognized for {config_tecnas.REPRESENTATION_TYPE}{config_tecnas.ENCODING_TYPE}' + Fore.RESET)
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
                mutated_child = self.change_block(self.children[i], mutation_type)

            else:
                print(Fore.LIGHTRED_EX + f'ERROR (mutate_children): {self.representation_type} Representation type not recognized' + Fore.RESET)
                return None
            print2(Fore.GREEN + f'MUTATING {self.children[i].idx} with {mutation_type} mutation complete\n')
            self.total_mut += 1
            mutated_child.idx = str(name_before_mut) + '[M]' #Add M to the index to indicate it was mutated      
            mutated_child = self.make_mutant(mutated_child, self.children[i])
            mutated_child = self.train_model(mutated_child)
            print2('Assesing if the mutation was succesful')
            #self.succ_mut += 1 if mutated_child.acc > self.children[i].acc else 0
            self.succ_mut += 1 if self.is_strictly_better(mutated_child, self.children[i]) else 0
            print2('Assesing if the mutation was succesful completed')
            self.children[i] = mutated_child
           
    def crossover(self, arch_obj1, arch_obj2):
        #     0                 1                  2                     3                      4                  5                    6              7                      8                   9               10
        #[{'INP': 32}, {'CONV': [128, 5]}, {'CONV': [128, 5]}, {'POOLMAX': [-1, 3]}, {'CONV': [64, 3]}, {'CONV': [256, 5]}, {'POOLMAX': [-1, 2]}, {'FLATTEN': None}, {'DENSE': [512, 'relu']}, {'LAST_DENSE': [10, 'softmax']}]
        #[5, 5, 9, 2, 7, 8, 12]
        #[C, C, P, C, C, P, D]
        crossover_obj = Crossover()
        #Check what kind of crossover is being used
        if self.crossover_type == 'SPC':
            point = random.choice(TECNAS.REP_CONSTANTS.SPC_INDEXES)
            gen_list_child1, gen_list_child2, idx1, idx2 = crossover_obj.single_point_crossover(arch_obj1, arch_obj2, point)
        elif self.crossover_type == 'TPC':
            [point1, point2] = random.sample(TECNAS.REP_CONSTANTS.SPC_INDEXES, 2)
            gen_list_child1, gen_list_child2, idx1, idx2 = crossover_obj.two_point_crossover(arch_obj1, arch_obj2, point1, point2)
        elif self.crossover_type == 'UC':
            gen_list_child1, gen_list_child2, idx1, idx2 = crossover_obj.uniform_crossover(arch_obj1, arch_obj2, crossover_indexes = TECNAS.REP_CONSTANTS.SPC_INDEXES)
        elif self.crossover_type == 'SBX':
            gen_list_child1, gen_list_child2, idx1, idx2, real_encoding_child1, real_encoding_child2 = crossover_obj.SBX(arch_obj1 = arch_obj1, arch_obj2 = arch_obj2, eta_c = 2.0, min_int = 0, max_int = 63)
            return crossover_obj.create_children(gen_list_child1 = gen_list_child1, gen_list_child2 = gen_list_child2, arch_obj1_idx = idx1, arch_obj2_idx = idx2, 
                                                 real_encoding_child1 = real_encoding_child1, real_encoding_child2 = real_encoding_child2)
        else:
            print(f'UNRECOGNIZED CROSSOVER METHOD {self.crossover_type}')
            return None, None
        return crossover_obj.create_children(gen_list_child1 = gen_list_child1, gen_list_child2 = gen_list_child2, arch_obj1_idx = idx1, arch_obj2_idx = idx2)
        

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
        self.nsga2_obj = TECNAS_NSGAII(pop = self.pop, pop_size = self.NPOP, tecnasObj = self) if config_tecnas.NSGA2 == True else None
        nsga2_parents_list = self.nsga2_obj.select_parent_pairs() if config_tecnas.NSGA2 == True else None
        
        for i in range(self.NPOP//2): #//2 because you add two children
            print2('\nSelecting parents')
            parent1, parent2 = self.random_parent_selection() if config_tecnas.NSGA2 == False else nsga2_parents_list[i]
            print2(Fore.LIGHTBLUE_EX + f'Using crossover')
            child1, child2 = self.crossover(parent1, parent2)
            child1.genotype.gen_list = self.validate_architecture(child1.genotype.gen_list)
            child2.genotype.gen_list = self.validate_architecture(child2.genotype.gen_list)
            child1.set_genoStr()
            child2.set_genoStr()
            child1 = self.make_child(child1, parent1, parent2)
            child2 = self.make_child(child2, parent1, parent2)
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
            arch_obj = BlockRepresentation(idx = str(idx), genotypeObj=genotype)
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
            arch_obj = BlockRepresentation(idx = str(idx), genotypeObj = genotype)
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
                print1(Fore.LIGHTBLUE_EX + f'\n{ast_bar} EXECUTION {self.exec}/{ConfigClass.EXECUTIONS} GENERATION {self.generation}/{ConfigClass.GENERATIONS} {self.crossover_type}_{self.mutation_type}  | NSGA2 {config_tecnas.NSGA2} | HHSE {self.HHSE} {hhse_str} {ast_bar}' + Fore.RESET)
            print1(Fore.YELLOW + f'{ConfigClass.DATASET} {ConfigClass.DATASET_PART*100}% |Repr Encoding: {self.representation_type}-{self.encoding_type}| |Surrogate: {self.SURROGATE}| |Train: {self.TRAIN}| |Simulate: {self.SIMULATE}| Surrogate folder: {self.experiment_folder}' + Fore.RESET)
            print1(Fore.YELLOW + f'Mutation prob: {self.MUT_PROB*100:.2f}%' + Fore.RESET)
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
        if ConfigClass.REPRESENTATION_TYPE == 'B': #
            arch.genotype.gen_list = BlockRepresentation.reset_add_pools(arch.genotype.gen_list)
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
    
    def set_arch_statuses(self):
        #Identify BEST, WORST AND MEDIAN. Save their info and reset their status. Also, calculate penalization for BEST.
        #Saving info about the WORST and MEDIAN ARCHITECTURES (BEFORE ELITISM) -------------------------- CHECKING THIS
        #Also store the worst arch and median
        print2('\nIdentifying BEST, MEDIAN and WORST architectures')
        self.pop[-1].archStatus = 'WORST'
        lower_median_idx = self.get_lower_median_architecture_idx(self.pop)
        self.pop[lower_median_idx].archStatus = 'MEDIAN'
        self.pop[0].archStatus = 'BEST'

        if self.generation > 1:
            self.pop[0].dPB = hamming_distance(self.prev_best_integerEnc, self.pop[0].integer_encoding)/len(self.pop[0].integer_encoding)
        else:
            print2('First generation, no previous best architecture')
            self.pop[0].dPB = 0
        self.prev_best_integerEnc = self.pop[0].integer_encoding.copy()
        #self.reporter.save_arch_info(self, self.pop[-1]) #Worst
        #self.reporter.save_arch_info(self, self.pop[lower_median_idx]) #Median
        self.pop[-1].archStatus = ''
        self.pop[lower_median_idx].archStatus = ''
        self.get_NFHT(self.pop[0])
        self.pop[0].flops, self.pop[0].num_params, self.pop[0].sizeMB = self.flops_params_gen_list_str(get_realMetrics = False, gen_list_str = str(self.pop[0].genotype.gen_list))
        #resetInfo is used for resetting some info about the best arch. saveRatios is for saving succ cross and succ mut.
        
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
            self.set_arch_statuses()
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

    def load_markov_matrices(self):
        #Load the Markov matrices from the CSV files and return them as a dictionary of dataframes.
        print1(f'Loading Markov matrices from {self.matrix_folder}')
        for filename in os.listdir(self.matrix_folder):
            if filename.startswith("matrix_") and filename.endswith(".csv"):
                file_path = os.path.join(self.matrix_folder, filename)
                df = pd.read_csv(file_path, index_col=0)
                self.markov_type = df.index.name
                self.markov_dict[self.markov_type] = df.copy()
        self.markov_type = ''
        print1('Markov matrices loaded successfully\n')

    def HHSE_MARKOV(self):
        progress = self.generation / self.total_gen
        if self.COMBINED_MARKOV: #COMBINED in HHSE_MARKOV_TYPES 
            if progress <= self.limits[0]:
                self.markov_type = 'MUT'
            elif progress <= self.limits[1]:
                self.markov_type = 'HV'
            elif progress <= self.limits[2]:
                self.markov_type = 'DHV'
            elif progress <= self.limits[3]:
                self.markov_type = 'CROSS'
            while self.current_state not in self.markov_dict[self.markov_type].index:
                print(f'Current state {self.current_state} not found in Markov matrix {self.markov_type}. Selecting another state randomly.')
                self.current_state = random.choice(self.markov_dict[self.markov_type].index.tolist())

        if self.generation == 1:
            self.current_state = random.choice(self.markov_dict[self.markov_type].index.tolist())
            self.GOs_history = [self.current_state]

        probabilities = self.markov_dict[self.markov_type][self.current_state].values
        states = self.markov_dict[self.markov_type].columns.tolist()
        probabilities = probabilities / probabilities.sum()
        next_state = np.random.choice(states, p=probabilities)
        return next_state

    def HHSE_Random(self):
        return random.choice(config_tecnas.crossoverList_hhse), random.choice(config_tecnas.mutationList_hhse)

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
            self.CROSS_GOs, self.MUT_GOs, self.HV_GOs = config_tecnas.set_HHSE_TEC_GOs(config_tecnas.REPRESENTATION_TYPE, config_tecnas.ENCODING_TYPE)
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
            self.CROSS_DICT, self.MUT_DICT, self.HV_DICT, self.GD_DICT = config_tecnas.set_HHSE_TEC_GOs(config_tecnas.REPRESENTATION_TYPE, config_tecnas.ENCODING_TYPE)
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
        self.prev_best_integerEnc = []
        
        for e in range(1,ConfigClass.EXECUTIONS+1):
            self.local_seed = config_tecnas.SEED_LIST[seed_idx]
            random.seed(self.local_seed)
            np.random.seed(self.local_seed)
            #tf.random.set_seed(self.local_seed)
            self.MUT_PROB = ConfigClass.MUT_PROB
            self.exec = e
            self.trained_archs = {}
            self.generation = 1
            self.initialize_pop()
            self.num_unchangedPB = 0
            self.op_count = {}
            self.GOs_history = []
            self.GOs_HV = {k: [0, 0, 0] for k in self.GOs_HV} #HV, PREV_HV and dHV.
            self.TECNAS_AOS_obj = TECNAS_AOS(tecnasObj = self)
            for g in range(1,ConfigClass.GENERATIONS+1):
                self.pool = []
                self.generation = g
                self.NSGA2_window() if config_tecnas.NSGA2_WINDOW  else None
                if self.HHSE == True: #Choose crossover and mutation type automatically
                    print2(Fore.MAGENTA + 'Hyper-heuristic selection of genetic operators' + Fore.RESET)
                    self.select_HHSE_method()
                if not config_tecnas.HHSE_GREEDY: #This means the new generation hasn't been generated yet. If this is true, GREEDY has already calculated the best self.pop
                    self.genetic_operators()
                    self.report_training = True
                    self.pool = self.pop + self.children
                    self.sorted_pool = sorted(self.pool, key=lambda obj: obj.acc, reverse = True) #Best archs first
                    self.pop = self.selection(pop = copy.deepcopy(self.pop)) #Here, WORST AND MEDIAN ARCHS ARE IDENTIFIED
                    self.GD = self.calculate_GD(self.pop)
                    self.reporter.save_generation_status(Status(tecnasObj = self, crosstype = self.crossover_type, muttype = self.mutation_type, pop = self.pop, JUST_PARETO = True)) #Save the info from the pareto front.
                    self.set_arch_statuses()
                if config_tecnas.HHSE_TEC:
                    if self.generation > 1:
                        self.TECNAS_AOS_obj.total_selections += 1 if config_tecnas.HHSE_TEC else 0
                        self.TECNAS_AOS_obj.update_arm(GO_str = f'{self.crossover_type}_{self.mutation_type}', reward = self.TECNAS_AOS_obj.calculate_reward(*self.AOS_rwd_percentages)) if config_tecnas.HHSE_TEC else None
                       
                    
            self.report_op_counts(results_folder = self.children[0].path_folder) if config_tecnas.HHSE_GREEDY_FIXED else None
            self.generate_total_counts(os.path.join(self.children[0].path_folder, f"counts_{config_tecnas.HHSE_GREEDY_CRITERIA}")) if config_tecnas.HHSE_GREEDY_FIXED else None
            self.reporter.flush(self.pop[0].path_filereport)
            seed_idx += 1 if seed_idx < len(config_tecnas.SEED_LIST)-1 else 0
            self.local_seed = config_tecnas.SEED_LIST[seed_idx]
            self.save_histories()
            self.markov_matrix(transitions = self.GOs_history, iterations=1000, tolerance=1e-10) if config_tecnas.HHSE_GREEDY == True else None
            self.TECNAS_AOS_obj.report_arms_csv(results_folder = self.children[0].path_folder) if config_tecnas.HHSE_TEC else None

        if config_tecnas.HHSE_GREEDY == True and not config_tecnas.HHSE_GREEDY_FIXED:
            self.markov_mean_stationary_probabilities()
            self.markov_barplot_mean_stationary_probabilities()
            self.markov_aggregate_matrix()

    
        

    def __init__(self, HHSE = False, NSGA2 = False, report_training = True, representation_type = ConfigClass.REPRESENTATION_TYPE, encoding_type = ConfigClass.ENCODING_TYPE, experiment_folder = '', regressor_type = -1):
        self.search_name = '' #Indicates type of search strategy and encoding. For example, LREAL_SPC_MPAR_NH, BREAL_MARKOV_MUT_NH (N means NSGA2, H means HHSE).
        self.report_training = report_training
        self.HHSE = HHSE #Hyper-heuristic selection of genetic operators
        self.NSGA2 = NSGA2
        self.ast_bar = 50*'+'
        self.representation_type = representation_type #Layer (L), Block (B)
        self.encoding_type = encoding_type
        self.NPOP = ConfigClass.MAIN_NPOP
        self.MUT_PROB = ConfigClass.MUT_PROB
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
        self.prev_best_integerEnc = None
        self.pop = []
        self.children = []
        self.pool = [] #Parents + children
        self.trained_parents = []
        self.flops_archs = {} #Dictionary with architectures whose FLOPs and NumParams were already calculated
        self.accuracy_histories = [] #To plot all accuracies through epochs
        self.loss_histories = [] #To plot all accuracies through epochs
        self.NUM_DENSE = ConfigClass.DATASET_NUMDENSE_DICT[ConfigClass.DATASET]
        self.MAXINT = Globals.INDEXES_POOLS[-1] if config_tecnas.REPRESENTATION_TYPE == 'L' else len(BlockRepresentation.all_blocks) - 1
        
        #============================================================= MAX MIN FLOPS AND PARAMS FOR HV NORMALIZATION ==============================================================
        self.BLOCKS_MINFLOPS  = 1.24e8
        self.BLOCKS_MAXFLOPS  = 1.6e10
        self.BLOCKS_MINPARAMS = 1.6e5
        self.BLOCKS_MAXPARAMS = 2.51e7

        self.LAYERS_MINFLOPS  = 1e6
        self.LAYERS_MAXFLOPS  = 1.71e10
        self.LAYERS_MINPARAMS = 1e3
        self.LAYERS_MAXPARAMS = 1.42e8

        self.MINFLOPS =  self.BLOCKS_MINFLOPS  if config_tecnas.REPRESENTATION_TYPE == 'B' else self.LAYERS_MINFLOPS
        self.MINPARAMS = self.BLOCKS_MINPARAMS if config_tecnas.REPRESENTATION_TYPE == 'B' else self.LAYERS_MINPARAMS
        self.MAXFLOPS =  self.BLOCKS_MAXFLOPS  if config_tecnas.REPRESENTATION_TYPE == 'B' else self.LAYERS_MAXFLOPS
        self.MAXPARAMS = self.BLOCKS_MAXPARAMS if config_tecnas.REPRESENTATION_TYPE == 'B' else self.LAYERS_MAXPARAMS

        #For HHSE ========================================= HHSE ========================================================================
        self.TECNAS_AOS_obj = None #For MAB and UCB.
        self.AOS_rwd_percentages = [0,0,0] #For calculating the reward in TECNAS AOS. [accuracy, GD, HV]
        #For GREEDY, only chose the identified GOs for each stage.
        self.GOs_names = [f"{cross}_{mut}" for cross, mut in product(self.crossover_types_list, self.mutation_types_list)] 
    
        if config_tecnas.IGNORE_RANDOM and 'NONE_NONE' in self.GOs_names:
            self.GOs_names.remove('NONE_NONE')
        #Save here all population generated through all combinations of crossover and mutation
        #For example, = {'SPC_MPAR': [arch1, arch2, ...], 'SPC_MSWAP': [arch1, arch2, ...], ...}
        self.GOs_names = sum(config_tecnas.set_HHSE_TEC_GOs(config_tecnas.REPRESENTATION_TYPE, config_tecnas.ENCODING_TYPE), []) if config_tecnas.HHSE_TEC else self.GOs_names
        self.GOs_population_dict = {}
        self.selected_by = ' ' #Depicts why the GO was selected as the best one in HHSE GREEDY
        self.GOs_history = [] #['SPC_MPAR', 'TPC_MSWAP', 'UC_MPAR', 'TPC_MSWAP',...} GOs order of application
        self.GOs_HV = {k: [0, 0, 0] for k in self.GOs_names} #HV, PREV_HV and dHV.
        #self.matrix_folder = os.path.join(path, 'results', experiment_folder, 'markov_matrix')
        self.matrix_folder = os.path.join(path, 'markov_matrices', f'markov_matrices_{config_tecnas.REPR_ENC_STR}')
        self.markov_dict = {}
        self.load_markov_matrices() if config_tecnas.HHSE_MARKOV == True else None
        self.loaded_markov_types = list(self.markov_dict.keys())
        self.markov_type = ''
        self.perc_intervals = config_tecnas.perc_intervals  #This is for setting the time intervals for COMBINED MARKOV.
        self.total_gen = config_tecnas.GENERATIONS
        self.limits = []
        self.cumulative = 0
        for p in self.perc_intervals:
                self.cumulative += p / 100
                self.limits.append(self.cumulative)

               
        #================================================= HHSE ========================================================================

        self.HV = 0 #Hypervolume of the pareto front in the current generation
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