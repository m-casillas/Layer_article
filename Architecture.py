from abc import ABC, abstractmethod
import random
from globalsENAS import *
from configENAS import *

class Architecture(ABC):
    def __str__(self):
        dec = 4
        ast = 50*'+'
        arch_info = [ast,
            f"Architecture ID: {self.idx}",
            #f"Type: {self.arch_type}",
            f"Accuracy: {self.acc:.4f}",
            f"Loss: {self.loss:.4f}",
            f"FLOPs: {self.flops}",
            f"CPU Hours/Min: {self.cpu_hours:.4f} hrs. or {self.cpu_hours*60:.4f} min.",
            f"Number of Parameters: {self.num_params}",
            f"Genotype: {self.genoStr}",
            ast+"\n"]
        return "\n".join(arch_info)
    
    def set_genotype(self, genotype):
        self.genotype = genotype #object of Genotype class
        self.genoStr = str(self.genotype.gen_list)

    def set_genoStr(self):
        self.genoStr = str(self.genotype.gen_list)
    
    def __init__(self, arch_type = 'S', idx = 9999, genotypeObj = None):
        self.arch_type = 'S' # S: Sequential
        self.idx = idx
        self.acc_hist = [random.randint(0,30) for i in range(EPOCHS)]
        self.loss_hist = [random.randint(0,10) for i in range(EPOCHS)]
        self.acc = random.randint(0,30)
        self.loss = random.randint(0,10)
        self.flops = random.randint(0,10)
        self.cpu_hours = random.randint(0,10)
        self.num_params = random.randint(0,10)
        self.genotype = genotypeObj #object of Genotype class
        self.genoStr = str(self.genotype.gen_list)
        self.parent1 = None
        self.parent2 = None
        self.before_mutation = None
        #Hamming distance between this architecture and the parent1 architecture (between encodings)
        self.dP1 = -1
        self.dP2 = -1
        self.dBM = -1
        self.isChild = False #Use this for the architecture info report
        self.isMutant = False
        self.wasInvalid = False
        self.bestGen = False #Best arch in the generation
        self.trained_epochs = EPOCHS
    
    @abstractmethod
    def decode(self):
        pass
    