from globalsENAS import *
from Architecture import *
from Genotype import *
from LayerRepresentation import *
from BlockRepresentation import *
"""#CROSSOVERR"""

#This class takes two architectures and combines them with Single Point Crossover
#(more to be adapted)

class Crossover:

    def create_children(self, gen_list_child1 = None, gen_list_child2 = None, arch_obj1_idx = None, arch_obj2_idx = None, real_encoding_child1 = None, real_encoding_child2 = None):
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

        if ConfigClass.REPRESENTATION_TYPE == 'REAL':
            child1_arch.real_encoding = real_encoding_child1
            child2_arch.real_encoding = real_encoding_child2

        return child1_arch, child2_arch

    def single_point_crossover(self, arch_obj1, arch_obj2, point):
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        gen_list_child1 = copy.deepcopy(self.parent1.genotype.gen_list[:point]) + copy.deepcopy(self.parent2.genotype.gen_list[point:])
        gen_list_child2 = copy.deepcopy(self.parent2.genotype.gen_list[:point]) + copy.deepcopy(self.parent1.genotype.gen_list[point:])
        return gen_list_child1, gen_list_child2, arch_obj1.idx, arch_obj2.idx

    def two_point_crossover(self, arch_obj1, arch_obj2, point1=2, point2=5):
        (point1, point2) = (point1, point2) if point1 < point2 else (point2, point1)
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        gen_list_child1 = (copy.deepcopy(self.parent1.genotype.gen_list[:point1]) +
                           copy.deepcopy(self.parent2.genotype.gen_list[point1:point2]) +
                           copy.deepcopy(self.parent1.genotype.gen_list[point2:]))
        gen_list_child2 = (copy.deepcopy(self.parent2.genotype.gen_list[:point1]) +
                           copy.deepcopy(self.parent1.genotype.gen_list[point1:point2]) +
                           copy.deepcopy(self.parent2.genotype.gen_list[point2:]))
        return gen_list_child1, gen_list_child2, arch_obj1.idx, arch_obj2.idx

    def uniform_crossover(self, arch_obj1, arch_obj2, crossover_indexes = []):
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        gen_list_child1 = copy.deepcopy(self.parent1.genotype.gen_list)
        gen_list_child2 = copy.deepcopy(self.parent2.genotype.gen_list)
        for i in crossover_indexes:
            if random.random() < 0.5:
                gen_list_child1[i], gen_list_child2[i] = gen_list_child2[i], gen_list_child1[i]
        return gen_list_child1, gen_list_child2, arch_obj1.idx, arch_obj2.idx
    
    def SBX(self, arch_obj1 = None, arch_obj2 = None, eta_c = 2.0, min_int = None, max_int = None):
        self.parent1 = arch_obj1
        self.parent2 = arch_obj2
        parent1_real = copy.deepcopy(arch_obj1.real_encoding)
        parent2_real = copy.deepcopy(arch_obj2.real_encoding)
        n = len(parent1_real)
        child1_real = [0.0] * n
        child2_real = [0.0] * n
        for i in range(n):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta_c + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
            child1_real[i] = 0.5 * ( (1 + beta) * parent1_real[i] + (1 - beta) * parent2_real[i])
            child2_real[i] = 0.5 * ( (1 - beta) * parent1_real[i] + (1 + beta) * parent2_real[i])
            child1_real[i] = check_within_bounds(child1_real[i], min_int, max_int)
            child2_real[i] = check_within_bounds(child2_real[i], min_int, max_int)
        #Create two temporal children to create the genlists. Transform the real encoding to integer encoding, then to genlist.
        child1_temp = copy.deepcopy(arch_obj1)
        child2_temp = copy.deepcopy(arch_obj2)
        child1_temp.real_encoding = child1_real
        child2_temp.real_encoding = child2_real
        child1_temp.real_encoding_to_integer_encoding()
        child2_temp.real_encoding_to_integer_encoding()
        gen_list_child1 = child1_temp.integer_vector_to_genList(child1_temp.integer_encoding)
        gen_list_child2 = child2_temp.integer_vector_to_genList(child2_temp.integer_encoding)
        return gen_list_child1, gen_list_child2, arch_obj1.idx, arch_obj2.idx, child1_real, child2_real

    def __init__(self):
        self.parent1 = None
        self.parent2 = None
