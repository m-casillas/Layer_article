import pandas as pd
class ReportENAS:
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
        pass