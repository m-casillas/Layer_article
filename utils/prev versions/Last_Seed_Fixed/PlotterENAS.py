from TECNAS import TECNAS
from globalsENAS import *



#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#genotype = Genotype('L', 'IV', gen_list)
#arch = Architecture('X', 0, 0, 0, 0, 1, genotype)

#

class Plotter:
    def get_median_oneCombination(self, df, cross, mut):
        median_trajectory_dict = {}
        cross_types = df['Crossover_type'].unique()
        mut_types = df['Mutation_type'].unique()
        last_generation = df['Generation'].max()
        print(f'\nMedians: Processing {cross} - {mut}')
        df_cross_mut = df[(df['Crossover_type'] == cross) & (df['Mutation_type'] == mut) & (df['arch_status'] == 'BEST')]
        df_last_gen = df_cross_mut[df_cross_mut['Generation'] == last_generation]
        #df_median = df_last_gen[df_last_gen['Accuracy'] == df_last_gen['Accuracy'].median()]
        df_sorted = df.sort_values(by="Accuracy").reset_index(drop=True)
        middle_index = len(df_sorted) // 2
        df_median = df_sorted.iloc[[middle_index]]
        df_median_row = df_median.iloc[0]
        median_exec = df_median_row['Execution'].astype(int)
        df_cross_mut_median_exec = df_cross_mut[df_cross_mut['Execution'] == median_exec]
        median_trajectory = df_cross_mut_median_exec['Accuracy']
        print(f'\nProcessing {cross}-{mut} completed')
        median_trajectory_dict[f'{cross}-{mut}'] = median_trajectory.tolist() #Saves combination and its median trajectory for the detected execution 
        #print(median_trajectory_dict)
        return median_trajectory_dict

    def get_all_medians_folder(self): 
        self.medians_folder = os.path.join(self.plot_folder, 'medians')
        folder = self.splitted_folder
        ensure_folder_exists(self.medians_folder)
        
        # Create a single figure
        plt.figure(figsize=(12, 6))
        
        for archs_file in os.listdir(folder):
            if archs_file.endswith(".csv"):
                
                print('Calculating median')
                print(f'Loading {archs_file}')
                filepath = os.path.join(folder, archs_file)
                df = pd.read_csv(filepath, encoding='utf-8')
                cross_types = df['Crossover_type'].unique()
                mut_types = df['Mutation_type'].unique()    
                median_trajectories_all_combinations = []
                for cross in cross_types:
                    for mut in mut_types:
                        median_trajectories_all_combinations.append(self.get_median_oneCombination(df, cross, mut))
               
                # Plot median trajectories
                generations = df['Generation'].unique().astype(str)
                
                for comb_dict in median_trajectories_all_combinations:
                    for comb, trajectory in comb_dict.items():
                        combin = comb
                        combin += '_NSGA2' if is_HHSE_NSGA2(archs_file)[1] else ''
                        combin +=  '_HHSE' if is_HHSE_NSGA2(archs_file)[0] else ''
                        plt.plot(generations, trajectory, label=combin)
        
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.title('Median Accuracy Trajectories for All Genetic Combinations')
        #plt.legend()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True)
        # Save the combined plot
        plot_path = os.path.join(self.medians_folder, 'combined_medians.png')
        #plt.savefig(plot_path)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)  # Increased DPI for clarity
        plt.clf()
        plt.close()
        print(f'Saving {plot_path} done')

    def boxplot_from_folder(self):
        #Get accuracy for all combinations and show their boxplots
        columns = ['Accuracy', 'FLOPs', 'Num_Params', 'HD_P1', 'HD_P2', 'HD_BM']
        #columns = ['HD_P1', 'HD_P2', 'HD_BM']
        measures = []
        labels = []
        for column in columns:
            measures = []
            labels = []
            for filename in os.listdir(self.splitted_folder):
                if filename.endswith(".csv"):
                    filepath = os.path.join(self.splitted_folder, filename)
                    print(f'\nBoxplots: Processing {filepath}')
                    df = pd.read_csv(filepath, encoding='utf-8')
                    dfBest = df[df['arch_status'] == 'BEST']
                    measures.append(dfBest[column])
                    combin = filename[:-4]  # Remove the .csv extension from the filename
                    labels.append(combin)
            plt.figure(figsize=(10, 6))
            plt.title(f'Boxplot of {column} for all combinations')
            #plt.boxplot(measures, tick_labels = labels)
            plt.boxplot(measures, labels = labels)
            plt.xticks(rotation=90, ha="right")
            plt.ylabel(column)
            plt.tight_layout()
            plot_path = os.path.join(self.boxplots, f'boxplot_{column}.png')
            print(f'Saving {plot_path}')
            plt.savefig(plot_path, dpi = 200)
            plt.clf()
            plt.close()
            print(f'Done\n')
        
      
    def plot_measures_from_folder(self, plot_bestarchs = True):
        """
        Reads all files (splitted) in the folder containing summarized results (bestarchs and GA performance), extracts mean and other performance values,
        and creates a bar plot showing those values. Used for comparing different crossover-mutation combinations.
        """
        self.barplot_folder = os.path.join(self.plot_folder, 'summarized_measures')
        if plot_bestarchs == True:
            self.to_be_plotted_folder = self.bestarchs_summaries_folder
            columns = self.columns_arch
        else:
            self.to_be_plotted_folder = self.GAs_summaries_folder
            columns = self.columns_GA
           
        ensure_folder_exists(self.barplot_folder)
        ensure_folder_exists(self.to_be_plotted_folder)

        for column in columns:
            all_accuracies = []
            labels = []
            for filename in os.listdir(self.to_be_plotted_folder):
                if filename.endswith(".csv"):
                    print(f'\nPlotting measures: Processing {filename}')
                    file_path = os.path.join(self.to_be_plotted_folder, filename)
                    df = pd.read_csv(file_path, encoding='utf-8')  
                    df_mean = df[df['Execution'] == 'Mean']
                    acc = df_mean[column].astype(float).iloc[0]
                    all_accuracies.append(acc)
                    #Get the cross and mut types from the filename
                    result = filename.replace("_bestarch_summary.csv", "") if plot_bestarchs else filename.replace("_GA_summary.csv", "") #Only get SPC_MWSAP, or TPC_MPAR_NSGA2
                    #result = parts.split("_")
                    #result += '_NSGA2' if is_HHSE_NSGA2(filename)[1] else ''
                    #result +=  '_HHSE' if is_HHSE_NSGA2(filename)[0] else ''
                    #result += df['HHSE_TYPE'].unique()[0]
                    labels.append(result)
                    #labels.append(f"{filename}_{column}")
                    print(f'Processing {filename} done')

            # Plot
            sorted_data = sorted(zip(all_accuracies, labels), reverse=True)
            all_accuracies, labels = zip(*sorted_data)
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, all_accuracies, color= ['forestgreen' if 'HHSE' in op else 'skyblue' for op in labels])
            plt.xlabel("Search strategy")
            plt.ylabel(column)
            plt.title(f"{column} per Strategy")
            plt.xticks(rotation=90, ha="right")
            # Add a small margin to y-limits for better visibility
            min_val = min(all_accuracies)
            max_val = max(all_accuracies)
            margin = (max_val - min_val) * 0.05 if max_val != min_val else 0.1
            #print(f'{min_val = }, {max_val = }, {margin = }')
            if np.isfinite(min_val) and np.isfinite(max_val):
                plt.ylim(min_val - margin, max_val + margin)
            else:
                print(f"Skipping y-limits for {column}: NaN or Inf detected")

            # Show y-ticks (default behavior)
            plt.yticks()
            # Add value labels on top of bars with 4 decimal places
            for bar in bars:
                height = bar.get_height()
                if height > 1e4:
                    sci = f'{height:.3e}'       # ejemplo: '1.23e+08'
                    mantissa, exp = sci.split('e')
                    label = f'{mantissa}e{int(exp)}' 
                else:
                    plt.text(bar.get_x() + bar.get_width() / 2, height + margin * 0.2,
                        f'{height:.4f}', ha='center', va='bottom')
                    
            plt.tight_layout()
            plot_path = os.path.join(self.barplot_folder, f'{column}.png')
            print(f'Saving {plot_path}')
            plt.savefig(plot_path, dpi = 200)
            plt.clf()
            plt.close()
            print(f'Done\n')

    def plot_convergence_exec(self):
        """
        Plot convergence curves per execution for multiple metrics.
        X-axis: Generations
        Y-axis: Metric value (Accuracy, HD_PB, HV, ...)
        One line per execution.
        """
        self.convergenceplots_bestarchs_folder = os.path.join(self.plot_folder, 'convergence_executions')
        ensure_folder_exists(self.convergenceplots_bestarchs_folder)
        folder = self.splitted_folder
        for filename in os.listdir(folder):
            if not filename.endswith(".csv"):
                continue
            filepath = os.path.join(folder, filename)
            print(f'\nConvergence plots: Processing {filepath}')
            df = pd.read_csv(filepath, encoding='utf-8')
            df_best = df[df['arch_status'] == 'BEST']
            exec_list = df_best['Execution'].unique()
            generation_list = sorted(df_best['Generation'].unique())
            generation_labels = [str(g) for g in generation_list]
            for column in config_tecnas.plot_convergency_columns:
                if column not in df_best.columns:
                    print(f'Column "{column}" not found in {filename}, skipping.')
                    continue
                plt.figure()
                for exec_id in exec_list:
                    df_exec = df_best[df_best['Execution'] == exec_id]
                    values_per_gen = []
                    for gen in generation_list:
                        df_gen = df_exec[df_exec['Generation'] == gen]
                        values_per_gen.append(df_gen[column].max())
                    plt.plot(generation_labels, values_per_gen, label=f'Execution {exec_id}')
                plt.xlabel('Generation')
                plt.ylabel(column)
                plt.title(f'Convergence of {column} per Generation\n{filename[:-4]}')
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plot_path = os.path.join(self.convergenceplots_bestarchs_folder, f'{filename[:-4]}_convergence_{column}.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=200)
                plt.clf()
                plt.close()
                print(f'Saving {plot_path} done')
        print('Convergence Plots... done')
   

    def plot_acc_loss_arch(self):
        #Reads archs from a CSV file and saves the convergence plots..
        self.convergenceplots_singlearchs_folder = os.path.join(self.plot_folder, 'convergence_individual_archs')
        ensure_folder_exists(self.convergenceplots_singlearchs_folder)
        plot_folder = self.convergenceplots_singlearchs_folder
        folder = self.experiment_folder
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder, filename)
                df = pd.read_csv(filepath, encoding='utf-8')
                arch_names = df['ID'].unique()
                for arch in arch_names:
                    print(f'Accuracy-Loss plots: Processing {arch}')
                    df_arch = df[df['ID'] == arch]
                    epochs_list = list(range(1, df_arch['Epochs'].iloc[0] + 1))
                    acc_list = ast.literal_eval(df_arch['Accuracy_history'].iloc[0]) #Convert strint to an actual list
                    loss_list = ast.literal_eval(df_arch['Loss_history'].iloc[0])
                    plt.plot(epochs_list, acc_list, label='Accuracy')
                    plt.plot(epochs_list, loss_list, label='Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy/Loss')
                    plt.title(f'Accuracy and Loss for Architecture {arch}')
                    plt.legend()
                    plot_path = os.path.join(plot_folder, f'{arch}.png')
                    plt.savefig(plot_path)
                    plt.clf()
                    plt.close()
                    print(f'Saving {plot_path} done')

    


    def plot_generation_status(self, raw_data = True, k_best = 15):
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.signal import savgol_filter
        # CONFIG
        sns.set_theme(style="whitegrid", context="talk", palette="tab10")
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        cmap = plt.cm.get_cmap('tab20', k_best)
        # HELPERS
        def compute_tendency(values):
            n = len(values)
            # ventana más agresiva
            window = max(5, int(0.2 * n))
            # savgol necesita número impar
            if window % 2 == 0:
                window += 1
            # fallback si hay pocos puntos
            if len(values) < window:
                return values
            return savgol_filter(values, window_length=window, polyorder=3)

        def find_best_point(values, generations, mode="max"):
            idx = np.nanargmax(values) if mode == "max" else np.nanargmin(values)
            return generations[idx], values[idx], idx

        # LOAD DATA
        columns = config_tecnas.plot_convergency_generation_status_columns
        self.generation_status_folder = os.path.join(self.plot_folder, 'generation_status')
        ensure_folder_exists(self.generation_status_folder)
        all_data = {}
        for filename in os.listdir(self.splitted_folder):
            if 'generation_status' not in filename:
                continue
            filepath = os.path.join(self.splitted_folder, filename)
            print(f'\nGeneration status plots: Loading {filepath}')
            df0 = pd.read_csv(filepath, encoding='utf-8')
            print(f'  Loaded {filename} with columns: {df0.columns.tolist()}')
            df = df0.groupby('Generation').mean(numeric_only=True).reset_index()
            stem = filename[:-4]
            all_data[stem] = {'df': df, 'df0': df0, 'generations': df["Generation"].values, 'search_name': df0["search_name"].iloc[0]}

        # PLOTS
        for column in columns:
            mode = columns[column][1]
            col_type = columns[column][0]
            operator_aucs = []
            # Compute AUC
            for stem, data in all_data.items():
                df = data['df']
                generations = data['generations']
                try:
                    if col_type == 'stats':
                        y_values = df[f'mean_{column}'].values
                    else:
                        y_values = df[column].values
                    auc = np.trapz(y_values, generations)
                    operator_aucs.append((auc, stem, data))
                except KeyError:
                    print(f'  Column {column} not found in {stem}, skipping.')
                    continue
            if not operator_aucs:
                continue

            # Rank by AUC
            operator_aucs.sort(key=lambda x: x[0], reverse=(mode == "max"))
            top_k = operator_aucs[:k_best]

            # FIGURE
            fig, ax = plt.subplots(figsize=(14, 8))
            for rank, (auc_val, stem, data) in enumerate(top_k):
                #color = cmap(rank)
                #linestyle = line_styles[rank % len(line_styles)]
                #marker = markers[rank % len(markers)]
                mutation_colors = {
                'MPOLY': '#1f77b4', 'MNUF': '#d62728', 'NONE':  '#2ca02c'}
                crossover_markers = {'SPC': 'o', 'TPC': 's', 'UC': '^', 'SBX': 'D', 'NONE': 'X'}
                crossover_linestyles = {'SPC': '-', 'TPC': '--', 'UC': '-.', 'SBX': ':', 'NONE': (0, (3,1,1,1))}
                parts = data['search_name'].split('_')
                if len(parts) >= 2:
                    crossover = parts[0]
                    mutation = parts[1]
                else:
                    crossover = 'NONE'
                    mutation = 'NONE'

                # Assign visual styles
                if crossover == 'NONE' and mutation == 'NONE':
                    color = 'black'
                else:
                    color = mutation_colors.get(mutation, 'black')
                marker = crossover_markers.get(crossover, 'o')
                linestyle = crossover_linestyles.get(crossover, '-')

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++
                df = data['df']
                generations = data['generations']
                search_name = data['search_name']
                # Values
                if col_type == 'stats':
                    mean_values = df[f'mean_{column}'].values
                    std_values = df[f'std_{column}'].values
                    y_values = mean_values
                    tendency = compute_tendency(mean_values)
                else:
                    y_values = df[column].values
                    tendency = compute_tendency(y_values)

                # Best point
                best_gen, best_val, _ = find_best_point(tendency,generations,mode)
                alpha = 0.9
                linewidth = 1.8
                # Seaborn Lineplot
                sns.lineplot(
                    x=generations,
                    y=tendency,
                    ax=ax,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    marker=marker,
                    markersize=5,
                    markevery=max(1, len(generations)//12),
                    label=f"[#{rank+1}] AUC={auc_val:.4f} | {search_name}"
                )
                # Best point marker
                ax.scatter(best_gen,best_val,color=color,s=120,marker='X',zorder=5)
            # STYLING
            ax.set_title(f'Top-{k_best} Operators by AUC — {column}', fontsize=18, pad=20)
            ax.set_xlabel("Generation", fontsize=14)
            ax.set_ylabel(column, fontsize=14)
            # ticks X
            all_gens = np.concatenate([data['generations'] for _, _, data in top_k] )
            ticks = [1] + list(np.arange(20, max(all_gens) + 1, 20))
            ax.set_xticks(ticks)
            ax.grid(True,linestyle='--', alpha=0.3)
            ax.legend(loc='best', fontsize=9, frameon=True, framealpha=0.8)
            plt.tight_layout( )
            # SAVE
            kbest_folder = os.path.join(self.generation_status_folder, f'top{k_best}_by_auc')
            ensure_folder_exists(kbest_folder)
            figname = os.path.join(kbest_folder, f'top{k_best}_{column}.png' )
            plt.savefig(figname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  Saved top-{k_best} AUC plot for {column} → {figname}')
        return
     
    def plot_generation_status2(self, k_best=10):
        def compute_tendency(values, window=5):
            n = len(values)
            window = max(3, int(0.1 * n))
            tendency = pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()
            return tendency

        def find_best_point(values, generations, mode="max"):
            idx = np.nanargmax(values) if mode == "max" else np.nanargmin(values)
            return generations[idx], values[idx], idx

        columns = config_tecnas.plot_convergency_generation_status_columns
        self.generation_status_folder = os.path.join(self.plot_folder, 'generation_status')
        ensure_folder_exists(self.generation_status_folder)

        # --- Step 1: Load all generation_status files ---
        all_data = {}  # {filename_stem: {'df': df, 'df0': df0, 'generations': ..., 'search_name': ...}}
        for filename in os.listdir(self.splitted_folder):
            if 'generation_status' in filename:
                filepath = os.path.join(self.splitted_folder, filename)
                print(f'\nGeneration status plots: Loading {filepath}')
                df0 = pd.read_csv(filepath, encoding='utf-8')
                print(f'  Loaded {filename} with columns: {df0.columns.tolist()}')
                df = df0.groupby('Generation').mean(numeric_only=True).reset_index()
                stem = filename[:-4]
                all_data[stem] = {
                    'df': df,
                    'df0': df0,
                    'generations': df["Generation"].values,
                    'search_name': df0["search_name"].iloc[0],
                }

        k_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        # --- Step 2: For each metric, rank operators by global AUC and plot k-best together ---
        for column in columns:
            mode = columns[column][1]
            col_type = columns[column][0]

            # Compute global AUC per operator for this column
            operator_aucs = []
            for stem, data in all_data.items():
                df = data['df']
                generations = data['generations']
                try:
                    if col_type == 'stats':
                        y_values = df[f'mean_{column}'].values
                    else:
                        y_values = df[column].values
                    auc = np.trapz(y_values, generations)
                    operator_aucs.append((auc, stem, data))
                except KeyError:
                    print(f'  Column {column} not found in {stem}, skipping.')
                    continue

            if not operator_aucs:
                continue

            # Rank by AUC (descending for max, ascending for min)
            operator_aucs.sort(key=lambda x: x[0], reverse=(mode == "max"))
            top_k = operator_aucs[:k_best]

            # --- Plot all k-best operators in one figure ---
            fig, ax = plt.subplots(figsize=(10, 6))
            for rank, (auc_val, stem, data) in enumerate(top_k):
                color = k_colors[rank % len(k_colors)]
                df = data['df']
                generations = data['generations']
                search_name = data['search_name']

                if col_type == 'stats':
                    mean_values = df[f'mean_{column}'].values
                    std_values = df[f'std_{column}'].values
                    y_values = mean_values
                    tendency = compute_tendency(mean_values)
                    ax.errorbar(generations, mean_values, yerr=std_values, fmt='o', color=color,
                                ecolor=color, capsize=4, elinewidth=1, alpha=0.4, zorder=1)
                    ax.fill_between(generations, mean_values - std_values, mean_values + std_values,
                                    alpha=0.1, color=color)
                else:
                    y_values = df[column].values
                    tendency = compute_tendency(y_values)
                    ax.scatter(generations, y_values, marker='o', color=color, alpha=0.4, zorder=1)

                best_gen, best_val, _ = find_best_point(tendency, generations, mode)
                ax.plot(generations, tendency, '--', color=color, linewidth=1,
                        label=f"[#{rank+1} AUC={auc_val:.4f}] {search_name}", zorder=3)
                ax.scatter(best_gen, best_val, color=color, marker='x', s=200, zorder=4)
                ax.axvline(best_gen, linestyle=':', color=color, alpha=0.5, zorder=0)
                ax.axhline(best_val, linestyle=':', color=color, alpha=0.5, zorder=0)

            ax.set_title(f'Top-{k_best} Operators by AUC — {column}')
            ax.set_xlabel("Generation")
            ax.set_ylabel(column)

            all_gens = np.concatenate([data['generations'] for _, _, data in top_k])
            ax.set_xticks(np.arange(min(all_gens), max(all_gens) + 1, 4))
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
            plt.tight_layout(rect=[0, 0, 0.8, 1])

            # Save in a dedicated k-best folder per metric
            kbest_folder = os.path.join(self.generation_status_folder, f'top{k_best}_by_auc')
            ensure_folder_exists(kbest_folder)
            figname = os.path.join(kbest_folder, f'top{k_best}_{column}.png')
            plt.savefig(figname, dpi=300)
            plt.close()
            print(f'  Saved top-{k_best} AUC plot for {column} → {figname}')
        '''
        # --- Step 3: Also produce individual plots as before ---
        for stem, data in all_data.items():
            df = data['df']
            df0 = data['df0']
            generations = data['generations']
            for column in columns:
                col_type = columns[column][0]
                mode = columns[column][1]
                plt.figure(figsize=(8, 6))
                if col_type == 'stats':
                    mean_values = df[f'mean_{column}'].values
                    std_values = df[f'std_{column}'].values
                    tendency = compute_tendency(mean_values)
                    plt.errorbar(generations, mean_values, yerr=std_values, fmt='o', color='blue',
                                ecolor='lightcoral', capsize=4, elinewidth=1, alpha=0.6,
                                label=f"Mean {column} with Std Dev", zorder=1)
                    y_min = np.nanmin(mean_values - std_values)
                    y_max = np.nanmax(mean_values + std_values)
                    plt.ylim(y_min, y_max)
                    y_values = mean_values
                else:
                    values = df[column].values
                    tendency = compute_tendency(values)
                    plt.scatter(generations, values, marker='o', color='blue', alpha=0.6,
                                label=f"{column} per Generation", zorder=1)
                    y_values = values

                auc = np.trapz(y_values, generations)
                best_gen, best_val, _ = find_best_point(tendency, generations, mode)
                plt.plot(generations, tendency, 'k--', linewidth=3, label="Tendency", zorder=3)
                plt.scatter(best_gen, best_val, color='red', marker='x', s=200,
                            label=f"Best {column}", zorder=4)
                plt.axvline(best_gen, linestyle='--', color='black', zorder=0)
                plt.axhline(best_val, linestyle='--', color='black', zorder=0)
                plt.title(f'{data["search_name"]} - {column} per Generation')
                plt.xlabel("Generation")
                plt.ylabel(column)
                plt.xticks(np.arange(min(generations), max(generations) + 1, 4))
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.text(0.02, 0.95, f"AUC = {auc:.4f}", transform=plt.gca().transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
                plt.legend()
                plt.tight_layout()

                fig_folder_per_name = os.path.join(self.generation_status_folder, stem)
                ensure_folder_exists(fig_folder_per_name)
                plt.savefig(os.path.join(fig_folder_per_name, f'{stem}_{column}.png'), dpi=300)
                fig_folder_per_metric = os.path.join(self.generation_status_folder, column)
                ensure_folder_exists(fig_folder_per_metric)
                plotname = os.path.join(fig_folder_per_metric, f'{stem}_{column}.png')
                print(f'Saving {plotname}')
                plt.savefig(plotname, dpi=300)
                plt.close()
        '''
        print('Generation Status Plots... done')

    def __init__(self, experiment_folder):
        os.system("cls")
        self.experiment_folder = experiment_folder
        self.bestarchs_summaries_folder = os.path.join(self.experiment_folder, 'bestarchs_summaries')
        self.GAs_summaries_folder = os.path.join(self.experiment_folder, 'GAs_summaries')
        self.plot_folder = os.path.join(self.experiment_folder, 'plots')
        self.splitted_folder = os.path.join(self.experiment_folder, 'splitted')
        self.boxplots = os.path.join(self.plot_folder, 'boxplots')
        ensure_folder_exists(self.experiment_folder)
        ensure_folder_exists(self.plot_folder)
        ensure_folder_exists(self.bestarchs_summaries_folder)
        ensure_folder_exists(self.GAs_summaries_folder)
        ensure_folder_exists(self.splitted_folder)
        ensure_folder_exists(self.boxplots)
        self.columns_arch = config_tecnas.plot_archcolumns
        self.columns_GA = config_tecnas.plot_GAcolumns
        


