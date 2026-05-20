#This class saves information about the current generation, in order to identify what genetic operators are more suitable.

class Status:
    def __init__(self, tecnasObj = None, crosstype = '', muttype = '', pop = None):
        self.pop = sorted(pop, key=lambda obj: obj.acc, reverse = True) #Best archs first
        self.tecnasObj = tecnasObj
        self.pop_size = len(tecnasObj.pop)
        self.crosstype = crosstype
        self.muttype = muttype
        self.GO_name = f'{self.crosstype}_{self.muttype}'
        self.path_report = tecnasObj.pop[0].path_folder
        self.update_status()
        self.GOs_history = {} #{'dPB': 'SPC_MPAR', 'TPC_MSWAP, 'UC_MSWAP', 'dHV': 'SPC_MPAR', 'TPC_MSWA',...}
        self.generation = self.tecnasObj.generation
        self.selected_by = tecnasObj.selected_by
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
                'GOs_history': [self.GOs_history]
        }
        
    def __str__(self):
        return f'{self.GO_name}, dHV: {self.dHV:.4f}, Current dPB: {self.dPB:.4f},  Succ Cross: {self.succ_cross:.4f}, Succ Mut: {self.succ_mut:.4f}'

    def update_status(self):
        self.exec = self.tecnasObj.exec
        self.generation = self.tecnasObj.generation
        self.mean_accuracy = sum(arch.acc for arch in self.pop)/self.pop_size
        self.mean_dP1 = sum(arch.dP1 for arch in self.pop)/self.pop_size
        self.mean_dP2 = sum(arch.dP2 for arch in self.pop)/self.pop_size
        self.mean_dBM = sum(arch.dBM for arch in self.pop)/self.pop_size
        self.dPB = self.pop[0].dPB
        self.best_accuracy = max(arch.acc for arch in self.pop)
        self.HV = self.tecnasObj.HV
        self.dHV = self.tecnasObj.dHV
        self.succ_cross = self.tecnasObj.succ_cross/self.tecnasObj.total_cross if self.tecnasObj.total_cross > 0 else 0
        self.succ_mut = self.tecnasObj.succ_mut/self.tecnasObj.total_mut if self.tecnasObj.total_mut > 0 else 0
        
        
                