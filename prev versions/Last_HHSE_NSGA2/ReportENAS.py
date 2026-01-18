from globalsENAS import *

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
               
        print(df)
        for col in all_columns[1:]: #Skip execution
            if col in selected_columns:
                print(f'19 {col = } {selected_columns = }')
                print(df[col])
                print()
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

    def save_arch_info(self, tecnasObj, arch = None, resetInfo = False, epochs = ConfigClass.EPOCHS):
        
        self.path_report = arch.path_folder
        if tecnasObj.REPORT_ARCH == False:
            print('REPORT_ARCH is False. No information will be saved in the architectures folder.')
            return

        succ_cross_ratio = 0 if tecnasObj.total_cross == 0 else tecnasObj.succ_cross/tecnasObj.total_cross
        succ_mut_ratio   = 0 if tecnasObj.total_mut == 0 else tecnasObj.succ_mut/tecnasObj.total_mut
        tecnasObj.crossover_type, tecnasObj.mutation_type = ('NONE', 'NONE') if tecnasObj.search_strategy == 'RANDOM' else (tecnasObj.crossover_type, tecnasObj.mutation_type)
        
        dict_archinfo = {'ID': [tecnasObj.representation_type + str(arch.idx)], 
             'Seed': [tecnasObj.local_seed],
             'Execution': [tecnasObj.exec],
             'Generation': [tecnasObj.generation],
             'GenCreation': [arch.gen_creation],
             'isChild': [arch.isChild],
             'isMutant': [arch.isMutant],
             'Integer_encoding': [str(arch.integer_encoding)],
             'Binary_encoding': [str(arch.binary_encoding)],
             'Genotype': [str(arch.genotype.gen_list)], 
             'Crossover_type': [tecnasObj.crossover_type if tecnasObj.HHSE == False else 'HHSE'],
             'Mutation_type': [tecnasObj.mutation_type if tecnasObj.HHSE == False else config_tecnas.NSGA_II_SELECTION_CSVREPORT[config_tecnas.NSGA_II_SELECTION[0]]],
             'Search_strategy': [tecnasObj.search_strategy],
             #'Type': [arch.arch_type], 
             'Epochs':[len(arch.acc_hist)],
             'Accuracy_history': [arch.acc_hist],
             'Loss_history': [arch.loss_hist],
             'Accuracy': [arch.acc], 
             'Top1': [arch.top1],
             'Top5': [arch.top5],
             'Loss': [arch.loss], 
             'FLOPs': [arch.flops], 
             'Num_Params': [arch.num_params], 
             'SizeMB': [arch.sizeMB],
             'P1_idx': [arch.P1Idx],
             'P2_idx': [arch.P2Idx],
             'P1': [arch.P1IntegerEncoding],
             'P2': [arch.P2IntegerEncoding],
             'Before_Mut': [arch.before_mutationIntegerEndcoding],
             'HD_P1': [arch.dP1],
             'HD_P2': [arch.dP2],
             'HD_BM': [arch.dBM],
             'Succ_Crossover': [tecnasObj.succ_cross],
             'Succ_Mutation': [tecnasObj.succ_mut],
             'Total_Crossovers': [tecnasObj.total_cross],
             'Total_Mutations': [tecnasObj.total_mut],
             'Succ_Crossover_ratio': [succ_cross_ratio if tecnasObj.HHSE == False else arch.succ_cross_ratio], #Because in HSSE, ratios are reseted.
             'Succ_Mutation_ratio': [succ_mut_ratio if tecnasObj.HHSE == False else arch.succ_mut_ratio],
             'Was_invalid': [arch.wasInvalid],
             'arch_status': [arch.archStatus],
             'NFHT': [arch.NFHT],
             'Trained_Completely':[False],
             'Has_CM':[False],
             'Confusion_matrix': [str(arch.cm)],
             'cm_accuracy': [arch.cm_accuracy],
             'cm_precision_macro': [arch.cm_precision_macro],
             'cm_recall_macro': [arch.cm_recall_macro],
             'cm_f1_macro': [arch.cm_f1_macro],
             'Ranking': [''],
             'HHSE': [tecnasObj.HHSE],
             'Objectives': [str(tecnasObj.objective_maxmin_names) if tecnasObj.HHSE == True else '']
            }

        if config_tecnas.FULL_REPORT == True:
            arch_info = dict_archinfo
        else:
            remove_columns = ['Loss_history', 'CPU_Sec', 'P1_idx', 'P2_idx', 'P1', 'P2', 'Before_Mut', 'Succ_Crossover', 'Succ_Mutation', 'Total_Crossovers', 'Total_Mutations', 'Was_invalid', 'Confusion_matrix']
            arch_info = remove_keys(dict_archinfo, remove_columns)

        data_arch = pd.DataFrame(dict_archinfo)
        HHSE = 'HHSE' if tecnasObj.HHSE == True else ''
        
        #Check if the file exists to add the headers or not.
        file_exists = os.path.exists(arch.path_filereport)
        data_arch.to_csv(arch.path_filereport, mode='a', index=False, header=not file_exists)
        print(f'Architecture ' + Fore.GREEN + f'{arch.idx}' + Style.RESET_ALL + f' info saved to {self.path_report}')
        
    def summarize_indicators(self, input_folder = '', output_folder = '', archs_info = True):
        #archs_info = False means GA info
        #Create a report showing all the performance measures for the best architectures per execution (best of all generations)
        #File must have only one crossover and one mutation type
        for input_file in os.listdir(input_folder):
            filename = f'{input_file[:-4]}'
            if input_file.endswith('.csv'):
                filepath = os.path.join(input_folder, input_file)
                print(f'Preparing to summarize performance indicators... {archs_info = }')
                print(f'Loading {filepath}')
                columns = ['Execution'] + (self.arch_columns if archs_info == True else self.GA_columns)
                df = pd.read_csv(filepath, encoding='utf-8')
                exec_list = df['Execution'].unique()
                table = []
                #Add all best arch's performance indicators per execution
                for exec in exec_list:
                    #exec = 1
                    #Obtain a dafatrame per execution, only best archs
                    df_bestexec = df[(df['Execution'] == exec) & (df['arch_status'] == 'BEST') & (df['Generation'] == df['Generation'].max())]
                    row = [exec]
                    for column in columns[1:]: #Skip execution
                        row.append(df_bestexec[column].values[0])
                    table.append(row)
                #Add mean and std at the end
                table = self.calculate_mean_std(table, columns, True)
                df_final = pd.DataFrame(table, columns=columns)
                newfilename = filename + ('_bestarch_summary.csv' if archs_info == True else '_GA_summary.csv')
                #newfilefolder = filename + '_summarized'
                report_folder = os.path.join(output_folder)#, newfilefolder)
                report_path = os.path.join(report_folder, newfilename)
                ensure_folder_exists(report_folder)
                df_final.to_csv(report_path, index = False)
                print(f'Report saved as {report_path}')
                print('Done\n')

    def __init__(self):
        self.crossover_type = None
        self.mutation_type = None
        self.generation = None
        self.execution = None

        self.current_dir =  os.getcwd()  #'/content/drive/MyDrive/DCC/Research/First_article'
        self.dir_archs = 'architectures'
        self.path_results = os.path.join(path, self.dir_archs)
        self.path_report = '' #This is where the report will be saved
        self.arch_columns = config_tecnas.plot_archcolumns
        self.GA_columns = config_tecnas.plot_GAcolumns

        #self.csv_files_folder = 'csv_files'
        #self.reports_folder = 'reports'
        #self.full_path = os.path.join(self.current_dir, self.csv_files_folder, self.reports_folder)

os.system("cls")
#reporter = ReportENAS()
#reporter.summarize_archs_report_folder()
#reporter.summarize_GA_performance_report_folder()





