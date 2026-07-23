import math
import os
import csv
import random
from colorama import Fore

class TECNAS_AOS:
    #Implement a multi-armed bandit algorithm along with UCB.
    def __init__(self, tecnasObj = None, c = 1):
        self.tecnasObj = tecnasObj
        self.pop = tecnasObj.pop
        self.pop_size = len(self.pop)
        self.GOs_pool = tecnasObj.GOs_names #['SPC_MPOLY', 'TPC_MNUF']
        self.arms = {GO:{'Ntimes':0, 'avg_Rwd':0, 'UCB':0} for GO in self.GOs_pool} # {'SPC_MPOLY': [Ntimes, avg_Rwd, UCB]} Number of times the GO was selected and its average reward.
        self.total_selections = 0 #Number of operator selections so far.
        self.number_of_operators = len(self.GOs_pool)
        self.c = c #Exploration parameter for UCB.
        self.update_population_stats()

    def update_population_stats(self):
        self.mean_accuracy = sum(arch.acc for arch in self.tecnasObj.pop)/self.pop_size
        self.mean_flops = sum(arch.flops for arch in self.tecnasObj.pop)/self.pop_size
        self.mean_params = sum(arch.num_params for arch in self.tecnasObj.pop)/self.pop_size
        self.mean_flops = ((self.mean_flops - self.tecnasObj.BLOCKS_MINPARAMS) / (self.tecnasObj.BLOCKS_MAXFLOPS - self.tecnasObj.BLOCKS_MINPARAMS))
        self.mean_params = ((self.mean_params - self.tecnasObj.BLOCKS_MINPARAMS) / (self.tecnasObj.BLOCKS_MAXPARAMS - self.tecnasObj.BLOCKS_MINPARAMS))

 
    def report_arms_csv(self, results_folder = ''):
        print(Fore.LIGHTBLUE_EX + f"Reporting arms to {os.path.join(results_folder, f'arms_report_E{self.tecnasObj.exec}.csv')}")
        with open(os.path.join(results_folder, f"arms_report_E{self.tecnasObj.exec}.csv"), 'w', newline='') as csvfile:
            fieldnames = ['GO', 'Ntimes', 'avg_Rwd', 'UCB']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for GO in self.GOs_pool:
                writer.writerow({'GO': GO, 'Ntimes': self.arms[GO]['Ntimes'], 'avg_Rwd': self.arms[GO]['avg_Rwd'], 'UCB': self.arms[GO]['UCB']})
        print(Fore.LIGHTBLUE_EX + f"Arms report saved to {os.path.join(results_folder, f'arms_report_E{self.tecnasObj.exec}.csv')}")

    def update_list_GOs(self, new_GOs):
        self.GOs_pool = new_GOs
        self.number_of_operators = len(self.GOs_pool)


    def calculate_reward(self, porc_HV = 0, porc_succ_cross = 0, porc_succ_mut = 0):
        self.update_population_stats()
        #50% HV, 25% CROSS, 25% MUT.
        #30% normalized dHV, 10% succesful crossovers and 60% successful mutations.
        #30% normalized dHV, 60% succesful crossovers and 10% successful mutations.
        #return porc_HV*self.tecnasObj.dHV + porc_succ_cross*self.tecnasObj.succ_cross + porc_succ_mut*self.tecnasObj.succ_mut
        return 1.40*self.mean_accuracy - 0.20*self.mean_flops - 0.20*self.mean_params
    
    def UCB(self, GO_str):
        self.arms[GO_str]['UCB'] = self.arms[GO_str]['avg_Rwd'] + self.c * math.sqrt(math.log(self.total_selections)/self.arms[GO_str]['Ntimes'])

    def update_arm(self, GO_str = '', reward = 0):
        self.arms[GO_str]['Ntimes'] += 1
        self.arms[GO_str]['avg_Rwd'] = self.arms[GO_str]['avg_Rwd'] + (reward - self.arms[GO_str]['avg_Rwd']) / self.arms[GO_str]['Ntimes']
        self.UCB(GO_str)

    def select_operator(self, candidates = []):
        #candidates: ['SPC_MPOLY', 'TPC_MNUF'] etc. 
        #Select the operator with the highest UCB value.
        if self.total_selections < self.number_of_operators:
            idx = self.total_selections
            #Select each operator at least once.
            GO_str = self.GOs_pool[idx]
        else:
            if len(candidates) == 0:
                candidates = self.GOs_pool
            #Select the operator with the highest UCB value.
            #GO_str = max(self.arms, key=lambda GO: self.arms[GO]['UCB'])
            max_ucb = max(self.arms[GO]['UCB'] for GO in candidates)
            best_candidates = [GO for GO in candidates if self.arms[GO]['UCB'] == max_ucb]
            GO_str = random.choice(best_candidates)
        return GO_str
    
