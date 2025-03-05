from abc import ABC, abstractmethod
import random

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
    
    def __init__(self, arch_type = 'S', idx = 9999, genotype = None):
        self.arch_type = 'S' # S: Sequential
        self.idx = idx
        self.acc_hist = [random.randint(0,10) for i in range(EPOCHS)]
        self.loss_hist = [random.randint(0,10) for i in range(EPOCHS)]
        self.acc = random.randint(0,10)
        self.loss = random.randint(0,10)
        self.flops = random.randint(0,10)
        self.cpu_hours = random.randint(0,10)
        self.num_params = random.randint(0,10)
        self.genotype = genotype #object of Genotype class
        self.genoStr = str(self.genotype.gen_list)
    
    @abstractmethod
    def decode(self):
        pass
    