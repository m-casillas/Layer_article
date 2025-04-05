import pandas as pd
import numpy as np
from globalsENAS import *
from configENAS import *
#from TECNAS import *

class ReportENAS:
    def save_arch_info(self, arch, se, ct, mt, g, e, ls, epochs = EPOCHS):
        if REPORT_ARCH == False:
            print('REPORT_ARCH is False. No information will be saved.')
            return
        
        self.generation = g + 1
        self.execution = e + 1
        self.crossover_type = ct 
        self.mutation_type = mt
        self.search_strategy = se

        parent1_integer_encoding = []
        parent2_integer_encoding = []
        before_mutation_integer_encoding = []

        if arch.isChild == False: #It is parent
            pass
        else:
            if arch.isMutant == True:
                before_mutation_integer_encoding = str(arch.before_mutation.integer_encoding)
            else:
                parent1_integer_encoding = str(arch.parent1.integer_encoding)
                parent2_integer_encoding = str(arch.parent2.integer_encoding)

        if se == 'RANDOM':
            self.crossover_type = 'NONE'
            self.mutation_type = 'NONE'
            

        data_arch = pd.DataFrame(
            {'ID': [arch.idx], 
             'Seed': [ls],
             'Execution': [self.execution],
             'Generation': [self.generation],
             'isChild': [arch.isChild],
             'isMutant': [arch.isMutant],
             'Integer_encoding': [str(arch.integer_encoding)],
             'Genotype': [arch.genoStr], 
             'Crossover type': [self.crossover_type],
             'Mutation type': [self.mutation_type],
             'Search strategy': [self.search_strategy],
             #'Type': [arch.arch_type], 
             'Epochs':[epochs],
             'Accuracy': [arch.acc], 
             'Loss': [arch.loss], 
             'FLOPs': [arch.flops], 
             'CPU Sec': [arch.cpu_hours*3600], 
             'Number of Parameters': [arch.num_params], 
             'Parent1': [parent1_integer_encoding],
             'Parent2': [parent2_integer_encoding],
             'Before Mutation': [before_mutation_integer_encoding],
             'HD_P1': [arch.dP1],
             'HD_P2': [arch.dP2],
             'HD_BM': [arch.dBM]}
        )
        path_report = os.path.join(path_results, f'{architecture_csv_filename}')
        #Check if the file exists to add the headers or not.
        file_exists = os.path.exists(path_report)

        data_arch.to_csv(path_report, mode='a', index=False, header=not file_exists)
        print(f'Architecture {arch.idx} info saved to {path_report}')


    def create_report(self, reporting_single_arch = False, single_arch = None):
        execList = list(range(1,EXECUTIONS+1))
        epochsList = list(range(1, EPOCHS+1))

        if reporting_single_arch == True: #Print the median arch information
            empty_list = ['']*(len(epochsList)-1)
            data = pd.DataFrame({   f"Epochs":epochsList,  f"Best_accuracy": single_arch.acc_hist,
                                    f"Loss": single_arch.loss_hist,  f"Acc_mean": [np.mean(single_arch.acc_hist)]+empty_list,
                                    f"Loss_mean": [np.mean(single_arch.loss_hist)]+empty_list
                                })
            path_report = os.path.join(path_results, f'{self.filename}_MEDIAN.csv')
        else:
            empty_list = ['']*(len(execList)-1)
            data = pd.DataFrame({   f"Execution":execList,  f"Best_accuracy": self.best_acc_list,
                                    f"Loss": self.loss_list,  f'CPU_hrs':self.cpu_hours_list,
                                    f'Num_params':self.num_params_list, f'FLOPs':self.flops_list,
                                    f"Acc_mean": [np.mean(self.best_acc_list)]+empty_list, f"Loss_mean": [np.mean(self.loss_list)]+empty_list,
                                    f'CPU_hrs_mean':[np.mean(self.cpu_hours_list)]+empty_list, f'Num_params_mean':[np.mean(self.num_params_list)]+empty_list,
                                    f'FLOPs_mean':[np.mean(self.flops_list)]+empty_list, f"Acc_std": [np.std(self.best_acc_list)]+empty_list,
                                    f"Loss_std": [np.std(self.loss_list)]+empty_list, f'CPU_hrs_std':[np.std(self.cpu_hours_list)]+empty_list,
                                    f'Num_params_std':[np.std(self.num_params_list)]+empty_list, f'FLOPs_std':[np.std(self.flops_list)]+empty_list
                                })

            path_report = os.path.join(path_results, f'{self.filename}.csv')
        print(f'Report saved in {path_report}')
        data.to_csv(path_report, index=False)
    def __init__(self):
        self.crossover_type = None
        self.mutation_type = None
        self.generation = None
        self.execution = None