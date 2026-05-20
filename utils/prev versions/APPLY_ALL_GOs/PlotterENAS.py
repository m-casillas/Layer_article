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
            plt.boxplot(measures, tick_labels = labels)
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
        
