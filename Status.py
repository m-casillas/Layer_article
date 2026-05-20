#This class saves information about the current generation, in order to identify what genetic operators are more suitable.

import time

import config_tecnas


class Status:
    def __init__(self, tecnasObj = None, crosstype = '', muttype = '', pop = None, JUST_PARETO = False):
        self.pop = sorted(pop, key=lambda obj: obj.acc, reverse = True) #Best archs first
        self.tecnasObj = tecnasObj
        self.pop_size = len(tecnasObj.pop)
        self.crosstype = crosstype
        self.muttype = muttype
        self.GO_name = f'{self.crosstype}_{self.muttype}'
        self.path_report = tecnasObj.pop[0].path_folder
        self.GOs_history = [] #{'dPB': 'SPC_MPAR', 'TPC_MSWAP, 'UC_MSWAP', 'dHV': 'SPC_MPAR', 'TPC_MSWA',...}
        self.op_count = {}
        self.prev_HV = 0
        self.exec = self.tecnasObj.exec
        self.generation = self.tecnasObj.generation
        self.mean_accuracy = sum(arch.acc for arch in self.pop)/self.pop_size
        self.std_accuracy = (sum((arch.acc - self.mean_accuracy) ** 2 for arch in self.pop) / self.pop_size) ** 0.5
        self.mean_flops = sum(arch.flops for arch in self.pop)/self.pop_size
        self.std_flops = (sum((arch.flops - self.mean_flops) ** 2 for arch in self.pop) / self.pop_size) ** 0.5
        self.mean_params = sum(arch.num_params for arch in self.pop)/self.pop_size
        self.std_params = (sum((arch.num_params - self.mean_params) ** 2 for arch in self.pop) / self.pop_size) ** 0.5
        self.mean_dP1 = sum(arch.dP1 for arch in self.pop)/self.pop_size
        self.mean_dP2 = sum(arch.dP2 for arch in self.pop)/self.pop_size
        self.mean_dBM = sum(arch.dBM for arch in self.pop)/self.pop_size
        self.dPB = self.pop[0].dPB
        self.best_accuracy = max(arch.acc for arch in self.pop)
        self.succ_cross = self.tecnasObj.succ_cross
        self.succ_mut = self.tecnasObj.succ_mut
        self.GD = self.tecnasObj.GD
        self.search_name = self.tecnasObj.search_name

        if JUST_PARETO: #Save information of the architectures in the Pareto front, each generation. For NOT GREEDY techniques (why did I call it JUST_PARETO?)
            self.HV = self.tecnasObj.HV
            self.dHV = self.tecnasObj.dHV
        else:
            self.HV = self.tecnasObj.GOs_HV[self.GO_name][0]
            #self.dHV = self.tecnasObj.GOs_HV[self.GO_name][1]
            self.dHV = self.tecnasObj.GOs_HV[self.GO_name][2]
        self.prev_HV = self.HV
        
        self.succ_cross = self.tecnasObj.succ_cross/self.tecnasObj.total_cross if self.tecnasObj.total_cross > 0 else 0
        self.succ_mut = self.tecnasObj.succ_mut/self.tecnasObj.total_mut if self.tecnasObj.total_mut > 0 else 0
        self.generation = self.tecnasObj.generation
        self.selected_by = config_tecnas.HHSE_GREEDY_CRITERIA #self.tecnasObj.selected_by

        self.dict_archinfo = {'Execution': [self.exec],
                'Generation': [self.generation],
                'Crossover_type': [self.crosstype],
                'Mutation_type': [self.muttype],
                'mean_accuracy': [self.mean_accuracy],
                'std_accuracy': [self.std_accuracy],
                'mean_flops': [self.mean_flops],
                'std_flops': [self.std_flops],
                'mean_params': [self.mean_params],
                'std_params': [self.std_params],
                'HV': [self.HV],
                'dHV': [self.dHV],
                'succ_cross': [self.succ_cross],
                'succ_mut': [self.succ_mut],
                'GD': [self.GD],
                'search_name': [self.search_name]
        }

        '''
        self.dict_archinfo = {'Execution': [self.exec],
                'Generation': [self.generation],
                'Crossover_type': [self.crosstype],
                'Mutation_type': [self.muttype],
                'best_accuracy': [self.best_accuracy],
                'mean_accuracy': [self.mean_accuracy],
                'mean_dP1': [self.mean_dP1],
                'mean_dP2': [self.mean_dP2],
                'mean_dBM': [self.mean_dBM],
                'dPB': [self.dPB],
                'HV': [self.HV],
                'dHV': [self.dHV],
                'succ_cross': [self.succ_cross],
                'succ_mut': [self.succ_mut],
                'Chosen_operator': [f'{self.crosstype}_{self.muttype}'],
                'Selected_by': [self.selected_by],
                'GOs_history': [self.GOs_history],
                'GOs_count':[self.op_count]
        }
        '''

    def update_chosen_GO(self, GOs_history, op_count):
        self.dict_archinfo['GOs_history'] = [GOs_history]
        self.dict_archinfo['GOs_count'] = [op_count]

    def __str__(self):
        return f'{self.GO_name}, dHV: {self.dHV:.4f}, Current dPB: {self.dPB:.4f},  Succ Cross: {self.succ_cross:.4f}, Succ Mut: {self.succ_mut:.4f}'

        
        
                