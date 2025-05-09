from file_handler import ensure_folder_exists
from TECNAS import TECNAS
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ast


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
        print(f'\nProcessing {cross} - {mut}')
        df_cross_mut = df[(df['Crossover_type'] == cross) & (df['Mutation_type'] == mut) & (df['BestGen'] == True)]
        df_last_gen = df_cross_mut[df_cross_mut['Generation'] == last_generation]
        df_median = df_last_gen[df_last_gen['Accuracy'] == df_last_gen['Accuracy'].median()]
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
        folder = self.experiment_folder
        ensure_folder_exists(self.medians_folder)
        for archs_file in os.listdir(folder):
            if archs_file.endswith(".csv"):
                print('Calculating median')
                print(f'Loading {archs_file}')
                filepath = os.path.join(folder, archs_file)
                df = pd.read_csv(filepath, encoding='utf-8')
                cross_types = df['Crossover_type'].unique()
                mut_types = df['Mutation_type'].unique()    

                print(f'{cross_types=}, {mut_types=}')
                median_trajectories_all_combinations = []
                for cross in cross_types:
                    for mut in mut_types:
                        median_trajectories_all_combinations.append(self.get_median_oneCombination(df, cross, mut))
                        
                # Plot all median trajectories
                generations = df['Generation'].unique().astype(str)
                
                plt.figure(figsize=(10, 6))
                
                for comb_dict in median_trajectories_all_combinations:
                    for comb, trajectory in comb_dict.items():
                        plt.plot(generations, trajectory, label=comb)
                plt.xlabel('Generation')
                plt.ylabel('Accuracy')
                plt.title('Median Accuracy Trajectories for All Combinations')
                plt.legend()
                plt.grid()
                plot_path = os.path.join(self.medians_folder, f'{archs_file}_medians.png')
                plt.savefig(plot_path)
                #plt.show()
                plt.clf()
                plt.close()
                print(f'Saving {plot_path} done')

    def plot_measures_from_folder(self, columns):
        """
        Reads all files (splitted) in the folder containing summarized results (bestarchs and GA performance), extracts mean and other performance values,
        and creates a bar plot showing those values. Used for comparing different crossover-mutation combinations.

        """
        self.barplot_folder = os.path.join(self.plot_folder, 'summarized_measures')
        if 'Accuracy' in columns:
            self.to_be_plotted_folder = self.bestarchs_summaries_folder
        else:
            self.to_be_plotted_folder = self.GAs_summaries_folder
        ensure_folder_exists(self.barplot_folder)
        ensure_folder_exists(self.to_be_plotted_folder)
        for column in columns:
            all_accuracies = []
            labels = []
            for filename in os.listdir(self.to_be_plotted_folder):
                if filename.endswith(".csv"):
                    print(f'\nProcessing {filename}')
                    file_path = os.path.join(self.to_be_plotted_folder, filename)
                    df = pd.read_csv(file_path, encoding='utf-8')  
                    df_mean = df[df['Execution'] == 'Mean']
                    print(f'{column}')
                    acc = df_mean[column].astype(float).iloc[0]
                    all_accuracies.append(acc)
                    #labels.append(f"{filename.replace('_bestarch_summary.csv', '')}_{column}")
                    labels.append(f"{filename}_{column}")
                    print(f'Processing {filename} done')

            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(labels, all_accuracies, color='skyblue')
            plt.xlabel("Search strategy")
            plt.ylabel(column)
            plt.title(f"{column} per Strategy")
            plt.xticks(rotation=45, ha="right")
            factor = 0.1
            plt.ylim(min(all_accuracies) - factor, max(all_accuracies) + factor)

            plt.tight_layout()
            plot_path = os.path.join(self.barplot_folder, f'{column}.png')
            print(f'Saving {plot_path}')
            plt.savefig(plot_path)
            #plt.show()
            plt.clf()
            plt.close()
            print(f'Done\n')

    def plot_convergence_exec(self):
        #Plot convergence of accurracy. x-axis are generations. y-axis are accuracies. Lines are for each execution
        #Each arch that had the best acc per generation is taken from the CSV file. Files should be splitted.
        #Reads archs from a CSV file and saves the convergence plots.
        self.convergenceplots_bestarchs_folder = os.path.join(self.plot_folder, 'convergence_executions')
        ensure_folder_exists(self.convergenceplots_bestarchs_folder)
        plot_folder = self.convergenceplots_bestarchs_folder
        folder = self.splitted_folder
        print(folder)
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder, filename)
                print(f'\nProcessing {filepath}')
                df = pd.read_csv(filepath, encoding='utf-8')
                #df_mut = df[df['isMutant'] == True]
                df_bestacc = df[df['BestGen'] == True] 
                execList = df_bestacc['Execution'].unique()
                generationList =  df_bestacc['Generation'].unique()
                
                max_accuracies = []
                accuracies_table = []
                for exec in execList:
                    df_exec = df_bestacc[(df_bestacc['Execution'] == exec)]
                    for generation in generationList:
                        df_generation = df_exec[df_exec['Generation'] == generation]
                        max_acc = df_generation['Accuracy'].max() #This is the only accuracy, since there is only one best per generation.
                        max_accuracies.append(max_acc)
                    accuracies_table.append(max_accuracies)
                    max_accuracies = []
                generationList = [str(g) for g in generationList]
                for i in range(len(execList)):
                    plt.plot(generationList, accuracies_table[i], label=f'Execution {i+1}')
                plt.xlabel('Generation')
                plt.ylabel('Accuracy')
                plt.title(f'Convergence of Accuracy per Generation {filename[:-4]}')
                plt.legend()
                #plt.plot(generationList, max_accuracies)
                #plt.show()
                plot_figure_path = os.path.join(plot_folder, f'{os.path.basename(filename[:-4])}_convergence.png') 
                plt.savefig(plot_figure_path)
                plt.clf()
                plt.close()
                print(f'Saving {plot_figure_path} done')
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
                    print(f'Processing {arch}')
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
        ensure_folder_exists(self.experiment_folder)
        ensure_folder_exists(self.plot_folder)
        ensure_folder_exists(self.bestarchs_summaries_folder)
        ensure_folder_exists(self.GAs_summaries_folder)
        ensure_folder_exists(self.splitted_folder)
        self.columns_arch = ['Accuracy', 'Loss', 'FLOPs', 'CPU_Sec', 'Num_Params']
        self.columns_GA = ['HD_P1_mean', 'HD_P2_mean', 'HD_BM_mean', 'succ_crossover', 'succ_mutation']
        

#plotter = Plotter()
#plotter.plot_measures_from_folder(plotter.columns_arch)
#plotter.plot_measures_from_folder(plotter.columns_GA)
#plotter.plot_acc_loss_arch()
#plotter.plot_convergence_exec()
csv_path = r"C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\experiment_2025-04-23_12-25\archs_2025-05-08_16-24.csv"
experiment_folder = r"C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\experiment_2025-04-23_12-25"
#plotter = Plotter(experiment_folder)
#plotter.get_all_medians_folder()  #NO FUNCIONA PARA RANDOM