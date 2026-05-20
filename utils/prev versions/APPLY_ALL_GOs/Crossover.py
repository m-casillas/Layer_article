from globalsENAS import *
from Architecture import *
from Genotype import *
from LayerRepresentation import *
from BlockRepresentation import *
"""#CROSSOVERR"""

#This class takes two architectures and combines them with Single Point Crossover
#(more to be adapted)

class Crossover:

    def create_children(self, arch_obj1_idx, arch_obj2_idx, gen_list_child1, gen_list_child2):
        idx1 = arch_obj1_idx+'|'+arch_obj2_idx+'(1)'
        idx2 = arch_obj1_idx+'|'+arch_obj2_idx+'(2)'
        idx1 = Architecture.set_arch_name(idx1)
        idx2 = Architecture.set_arch_name(idx2)

        if ConfigClass.REPRESENTATION_TYPE == 'L':
            child1_arch = LayerRepresentation(idx = idx1, genotypeObj = Genotype('L', ConfigClass.ENCODING_TYPE, gen_list_child1))
            child2_arch = LayerRepresentation(idx = idx2, genotypeObj = Genotype('L', ConfigClass.ENCODING_TYPE, gen_list_child2))
        elif ConfigClass.REPRESENTATION_TYPE == 'B':
            child1_arch = BlockRepresentation(idx = idx1, genotypeObj = Genotype('B', ConfigClass.ENCODING_TYPE, gen_list_child1))
            child2_arch = BlockRepresentation(idx = idx2, genotypeObj = Genotype('B', ConfigClass.ENCODING_TYPE, gen_list_child2))

        return child1_arch, child2_arch

    def single_point_crossover(self, arch_obj1, arch_obj2, point):
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        #                  0             1              2                3                   4                    5                  6                    7
        #gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
        gen_list_child1 = self.parent1.genotype.gen_list[:point] + self.parent2.genotype.gen_list[point:]#len(self.parent2.genotype.gen_list)-3]
        gen_list_child2 = self.parent2.genotype.gen_list[:point] + self.parent1.genotype.gen_list[point:]#len(self.parent1.genotype.gen_list)-3]
        return self.create_children(arch_obj1.idx, arch_obj2.idx, gen_list_child1, gen_list_child2)
        

    def two_point_crossover(self, arch_obj1, arch_obj2, point1 = 2, point2 = 5):
        (point1, point2) = (point1, point2) if point1 < point2 else (point2, point1)
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        gen_list_child1 = (self.parent1.genotype.gen_list[:point1] + self.parent2.genotype.gen_list[point1:point2] + self.parent1.genotype.gen_list[point2:])
        gen_list_child2 = (self.parent2.genotype.gen_list[:point1] + self.parent1.genotype.gen_list[point1:point2] + self.parent2.genotype.gen_list[point2:])
        return self.create_children(arch_obj1.idx, arch_obj2.idx, gen_list_child1, gen_list_child2)
    
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
        return self.create_children(arch_obj1.idx, arch_obj2.idx, gen_list_child1, gen_list_child2)

    def __init__(self):
        self.parent1 = None
        self.parent2 = None
