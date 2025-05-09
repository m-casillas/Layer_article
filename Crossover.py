from globalsENAS import *
from Architecture import *
from Genotype import *
from LayerRepresentation import *
"""#CROSSOVERR"""

#This class takes two architectures and combines them with Single Point Crossover
#(more to be adapted)

class Crossover:
    def single_point_crossover(self, arch_obj1, arch_obj2, point):
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        #                  0             1              2                3                   4                    5                  6                    7
        #gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

        gen_list_child1 = self.parent1.genotype.gen_list[:point] + self.parent2.genotype.gen_list[point:]#len(self.parent2.genotype.gen_list)-3]
        gen_list_child2 = self.parent2.genotype.gen_list[:point] + self.parent1.genotype.gen_list[point:]#len(self.parent1.genotype.gen_list)-3]
        child1_arch = LayerRepresentation(arch_type = 'S', idx = arch_obj1.idx+'|'+arch_obj2.idx+'(1)', genotypeObj = Genotype('L', 'IV', gen_list_child1))
        child2_arch = LayerRepresentation(arch_type = 'S', idx = arch_obj1.idx+'|'+arch_obj2.idx+'(2)', genotypeObj = Genotype('L', 'IV', gen_list_child2))
        return child1_arch, child2_arch

    def two_point_crossover(self, arch_obj1, arch_obj2, point1 = 2, point2 = 5):
        (point1, point2) = (point1, point2) if point1 < point2 else (point2, point1)
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        gen_list_child1 = (self.parent1.genotype.gen_list[:point1] + self.parent2.genotype.gen_list[point1:point2] + self.parent1.genotype.gen_list[point2:])
        gen_list_child2 = (self.parent2.genotype.gen_list[:point1] + self.parent1.genotype.gen_list[point1:point2] + self.parent2.genotype.gen_list[point2:])
        child1_arch = LayerRepresentation(arch_type='S', idx=arch_obj1.idx + '-' + arch_obj2.idx + '(1)', genotypeObj=Genotype('L', 'IV', gen_list_child1))
        child2_arch = LayerRepresentation(arch_type='S', idx=arch_obj1.idx + '-' + arch_obj2.idx + '(2)', genotypeObj=Genotype('L', 'IV', gen_list_child2))
        return child1_arch, child2_arch
    
    def uniform_crossover(self, arch_obj1, arch_obj2):
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2

        gen_list_child1 = []
        gen_list_child2 = []

        for gene1, gene2 in zip(self.parent1.genotype.gen_list, self.parent2.genotype.gen_list):
            if random.random() < 0.5:
                gen_list_child1.append(gene1)
                gen_list_child2.append(gene2)
            else:
                gen_list_child1.append(gene2)
                gen_list_child2.append(gene1)

        child1_arch = LayerRepresentation(arch_type='S', idx=arch_obj1.idx + '-' + arch_obj2.idx + '(1)', genotypeObj=Genotype('L', 'IV', gen_list_child1))
        child2_arch = LayerRepresentation(arch_type='S', idx=arch_obj1.idx + '-' + arch_obj2.idx + '(2)', genotypeObj=Genotype('L', 'IV', gen_list_child2))

        return child1_arch, child2_arch

    def __init__(self):
        self.parent1 = None
        self.parent2 = None
