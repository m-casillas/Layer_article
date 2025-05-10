import pandas as pd
import numpy as np
from globalsENAS import *
from configENAS import *
from file_handler import ensure_folder_exists
#from TECNAS import *

class ReportENAS:
    def stats_select_columns(self, df, all_columns, selected_columns, stat):
        row_measures = []
        #stat = 'mean' or 'std'
        if stat == 'mean':
            row_measures.append('Mean')
            stat_func = np.mean
        elif stat == 'std':
            row_measures.append('Std')
            stat_func = np.std
        else:
            raise ValueError("Invalid statistic type. Use 'mean' or 'std'.")
               
        
        for col in all_columns[1:]:
            if col in selected_columns:
                row_measures.append(stat_func(df[col]))
            else:
                row_measures.append('')
        return pd.concat([df, pd.DataFrame([row_measures], columns=df.columns)], ignore_index=True)

    def calculate_mean_std(self, table, columns, add_label = False):
        #add_label decites if Mean and Std are added or not.
        #Add mean and std at the end
        table_np = np.array(table)
        row_measures = []
        if add_label == True:
            row_measures.append('Mean')
        
        for i in range(1,len(columns)):
            row_measures.append(np.mean(table_np[:, i]))
        table.append(row_measures)
        row_measures = []
        if add_label == True:
            row_measures.append('Std')
        for i in range(1,len(columns)):
            row_measures.append(np.std(table_np[:, i]))
        table.append(row_measures)
        return table

    def save_arch_info(self, tecnasObj, arch = None, isBest = False, epochs = EPOCHS):
        if REPORT_ARCH == False:
            print('REPORT_ARCH is False. No information will be saved.')
            return
        if isBest == True:
             local_seed = -999999
             succ_cross = -999999
             succ_mut = -999999
             total_cross = -999999
             total_mut = -999999
        else:
            local_seed = tecnasObj.local_seed
            succ_cross = tecnasObj.succ_cross
            succ_mut = tecnasObj.succ_mut
            total_cross = tecnasObj.total_cross
            total_mut = tecnasObj.total_mut

            

        parent1_integer_encoding = 'NONE'
        parent2_integer_encoding = 'NONE'
        before_mutation_integer_encoding = 'NONE'
        p1_idx = 'NONE'
        p2_idx = 'NONE'

        if arch.isChild == False: #It is parent
            pass
        else:
            if arch.isMutant == True:
                before_mutation_integer_encoding = str(arch.before_mutation.integer_encoding)
            else:
                p1_idx = arch.parent1.idx
                p2_idx = arch.parent2.idx
                parent1_integer_encoding = str(arch.parent1.integer_encoding)
                parent2_integer_encoding = str(arch.parent2.integer_encoding)

        if tecnasObj.search_strategy == 'RANDOM':
            tecnasObj.crossover_type = 'NONE'
            tecnasObj.mutation_type = 'NONE'
            

        data_arch = pd.DataFrame(
            {'ID': [arch.idx], 
             'Seed': [local_seed],
             'Execution': [tecnasObj.exec],
             'Generation': [tecnasObj.generation],
             'isChild': [arch.isChild],
             'isMutant': [arch.isMutant],
             'Integer_encoding': [str(arch.integer_encoding)],
             'Genotype': [arch.genoStr], 
             'Crossover_type': [tecnasObj.crossover_type],
             'Mutation_type': [tecnasObj.mutation_type],
             'Search_strategy': [tecnasObj.search_strategy],
             #'Type': [arch.arch_type], 
             'Epochs':[len(arch.acc_hist)],
             'Accuracy_history': [arch.acc_hist],
             'Loss_history': [arch.loss_hist],
             'Accuracy': [arch.acc], 
             'Loss': [arch.loss], 
             'FLOPs': [arch.flops], 
             'CPU_Sec': [arch.cpu_hours*3600], 
             'Num_Params': [arch.num_params], 
             'P1_idx': [p1_idx],
             'P2_idx': [p2_idx],
             'P1': [parent1_integer_encoding],
             'P2': [parent2_integer_encoding],
             'Before_Mut': [before_mutation_integer_encoding],
             'HD_P1': [arch.dP1],
             'HD_P2': [arch.dP2],
             'HD_BM': [arch.dBM],
             'Succ_Crossover': [succ_cross],
             'Succ_Mutation': [succ_mut],
             'Total_Crossovers': [total_cross],
             'Total_Mutations': [total_mut],
             'Was_invalid': [arch.wasInvalid],
             'BestGen': [arch.bestGen]
             }
        )
        ensure_folder_exists(self.path_results)
        path_report = os.path.join(self.path_results, f'{architecture_csv_filename}')
        #Check if the file exists to add the headers or not.
        file_exists = os.path.exists(path_report)
        data_arch.to_csv(path_report, mode='a', index=False, header=not file_exists)
        print(f'Architecture {arch.idx} info saved to {path_report}')

    def summarize_bestarchs_report(self, input_folder, output_folder):
        #Create a report showing all the performance measures for the best architectures per execution (best of all generations)
        #File must have only one crossover and one mutation type
        for input_file in os.listdir(input_folder):
            filename = f'{input_file[:-4]}'
            if input_file.endswith('.csv'):
                filepath = os.path.join(input_folder, input_file)
                print('Preparing to summarize best architecture performance...')
                print(f'Loading {filepath}')
                columns = ['Execution', 'Accuracy', 'Loss', 'FLOPs', 'CPU_Sec', 'Num_Params']
                df = pd.read_csv(filepath, encoding='utf-8')
                exec_list = df['Execution'].unique()
                table = []
                #Add all best arch's performance indicators per execution
                for exec in exec_list:
                    #exec = 1
                    #Obtain a dafatrame per execution, only mutants and only the last generation.
                    df_exec = df[(df['Execution'] == exec) & (df['BestGen'] == True)]
                    best_acc = df_exec['Accuracy'].max()
                    best_arch = df_exec[df_exec['Accuracy'] == best_acc]
                    loss_from_best = best_arch['Loss'].values[0]
                    flops_from_best = best_arch['FLOPs'].values[0]
                    cpu_hours_from_best = best_arch['CPU_Sec'].values[0]
                    num_params_from_best = best_arch['Num_Params'].values[0]
                    row = [exec, best_acc, loss_from_best, flops_from_best, cpu_hours_from_best, num_params_from_best]
                    table.append(row)
                #Add mean and std at the end
                table = self.calculate_mean_std(table, columns, True)
                df_final = pd.DataFrame(table, columns=columns)
                newfilename = filename + '_bestarch_summary.csv'
                #newfilefolder = filename + '_summarized'
                report_folder = os.path.join(output_folder)#, newfilefolder)
                report_path = os.path.join(report_folder, newfilename)
                ensure_folder_exists(report_folder)
                df_final.to_csv(report_path, index = False)
                print(f'Report saved as {report_path}')
                print('Done\n')

    def summarize_GA_report(self, input_folder, output_folder):
        #Create a report summarizing the GAs performance: distances and successful mutations and crossovers per execution
        #Create a table with non-mutated children and calculate distances between their parents
        #Also, add the information about distances of mutated children, and succesful mutations and crossovers
        #Create a report showing all the performance measures for the best architectures per execution
        
        for input_file in os.listdir(input_folder):
            filename = f'{input_file[:-4]}'
            if 'NONE' in filename:
                print(f'Skipping {filename} because is RANDOM or it has NONE crossover or mutation')
                continue
            if input_file.endswith('.csv'):
                filepath = os.path.join(input_folder, input_file)
                print('Preparing to summarize GA performance...')
                print(f'Loading {filepath}')
                
                #columns_file = ['Execution', 'HD_P1_mean', 'HD_P2_mean', 'HD_P1_std', 'HD_P2_std', 'HD_BF', 'HD_BF_mean', 'HD_BF_std', 'succ_crossover', 'succ_mutation']
                df = pd.read_csv(filepath, encoding='utf-8')
                exec_list = df['Execution'].unique()
                #For HD of parents ================================================================================
                table = []
                columns = ['Execution', 'HD_P1_mean', 'HD_P2_mean', 'HD_P1_std', 'HD_P2_std']
                for exec in exec_list:
                    df_children = df[(df['Execution'] == exec) & (df['isMutant'] == False) & (df['isChild'] == True) & (df['BestGen'] == False)]
                    HD_P1_mean = df_children['HD_P1'].mean()
                    HD_P2_mean = df_children['HD_P2'].mean()
                    HD_P1_std = df_children['HD_P1'].std()
                    HD_P2_std = df_children['HD_P2'].std()
                    row = [exec, HD_P1_mean, HD_P2_mean, HD_P1_std, HD_P2_std]
                    table.append(row)
                df_HD_Parents = pd.DataFrame(table, columns=columns)

                #For HD of before mutation ================================================================================
                table = []
                columns = ['HD_BM_mean', 'HD_BM_std']
                for exec in exec_list:
                    df_children = df[(df['Execution'] == exec) & (df['isMutant'] == True)]
                    HD_BF_mean = df_children['HD_BM'].mean()
                    HD_BF_std = df_children['HD_BM'].std()
                    row = [HD_BF_mean, HD_BF_std]
                    table.append(row)
                df_HD_BeforeMut = pd.DataFrame(table, columns=columns)
                
                #For succeful mutation and crossover. ================================================================================
                generations_list = df['Generation'].unique()
                table = []
                columns = ['succ_crossover', 'succ_mutation']
                #total_muts = df['Total_Mutations'].max()
                #total_cross = df['Total_Crossovers'].max()
                total_muts = 6*3
                total_cross = 3*3
                for exec in exec_list:
                    succ_crossover = 0
                    succ_mutation = 0
                    for gener in generations_list:
                        df_exec_gener = df[(df['Execution'] == exec) & (df['Generation'] == gener)]
                        succ_crossover += df_exec_gener['Succ_Crossover'].max()
                        succ_mutation += df_exec_gener['Succ_Mutation'].max()
                    row = [succ_crossover/total_muts, succ_mutation/total_cross]
                    table.append(row)
                df_succ_GAs = pd.DataFrame(table, columns=columns)
                #Join all dataframes
                df_final = pd.concat([df_HD_Parents, df_HD_BeforeMut, df_succ_GAs], axis=1)
                
                #For mean and std of only these columns
                columns = ['Execution', 'HD_P1_mean', 'HD_P2_mean', 'HD_P1_std', 'HD_P2_std','HD_BM_mean', 'HD_BM_std', 'succ_crossover', 'succ_mutation']
                columns_mean = ['HD_P1_mean', 'HD_P2_mean','HD_BM_mean','succ_crossover', 'succ_mutation']
                df_final = self.stats_select_columns(df_final, columns, columns_mean, 'mean')
                columns_std = ['HD_P1_mean', 'HD_P2_mean', 'HD_BM_mean', 'succ_crossover', 'succ_mutation']
                df_final = self.stats_select_columns(df_final, columns, columns_mean, 'std')
                newfilename = filename + '_GA_summary.csv'
                report_folder = os.path.join(output_folder)#, newfilefolder)
                report_path = os.path.join(report_folder, newfilename)
                ensure_folder_exists(report_folder)
                df_final.to_csv(report_path, index = False)
                print(f'Report saved as {report_path}')
                print('Done\n')

    
    def summarize_archs_report_folder(self):
        #Summarize the best architecture performance per execution
        to_be_bestarch_summarized_folder = 'to_be_bestarch_summarized'
        bestarch_summaries_folder = 'bestarch_summaries'
        to_be_bestarch_summarized_path = os.path.join(self.full_path, to_be_bestarch_summarized_folder)
        bestarch_summaries_path = os.path.join(to_be_bestarch_summarized_path, bestarch_summaries_folder)
        ensure_folder_exists(to_be_bestarch_summarized_path)
        ensure_folder_exists(bestarch_summaries_path)
        self.summarize_archs_report(to_be_bestarch_summarized_path, bestarch_summaries_path)
#
    
    def __init__(self):
        self.crossover_type = None
        self.mutation_type = None
        self.generation = None
        self.execution = None

        self.current_dir =  os.getcwd()  #'/content/drive/MyDrive/DCC/Research/First_article'
        self.dir_archs = 'architectures'
        self.path_results = os.path.join(path, self.dir_archs)


        self.csv_files_folder = 'csv_files'
        self.reports_folder = 'reports'
        self.full_path = os.path.join(self.current_dir, self.csv_files_folder, self.reports_folder)

os.system("cls")
#reporter = ReportENAS()
#reporter.summarize_archs_report_folder()
#reporter.summarize_GA_performance_report_folder()





