from globalsENAS import *
from Architecture import *
from Genotype import *
from LayerRepresentation import *
from BlockRepresentation import *
"""#CROSSOVERR"""

#This class takes two architectures and combines them with Single Point Crossover
#(more to be adapted)

class Crossover:

    def single_point_crossover(self, v1 = None, v2 = None, point = 9999):
        child1 = copy.deepcopy(v1[:point]) + copy.deepcopy(v2[point:])
        child2 = copy.deepcopy(v2[:point]) + copy.deepcopy(v1[point:])
        return child1, child2

    def two_point_crossover(self, v1 = None, v2 = None, point1 = 2, point2 = 5):
        (point1, point2) = (point1, point2) if point1 < point2 else (point2, point1)
        child1 = (copy.deepcopy(v1[:point1]) +
                  copy.deepcopy(v2[point1:point2]) +
                  copy.deepcopy(v1[point2:]))
        child2 = (copy.deepcopy(v2[:point1]) +
                  copy.deepcopy(v1[point1:point2]) +
                  copy.deepcopy(v2[point2:]))  
        return child1, child2

    def uniform_crossover(self, v1 = None, v2 = None, crossover_indexes = []):
        child1 = copy.deepcopy(v1)
        child2 = copy.deepcopy(v2)
        for i in crossover_indexes:
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def SBX(self, v1_real = None, v2_real = None, eta_c = 2.0, min_int = None, max_int = None, INTEGER_ENCODING2 = False):
        n = len(v1_real)
        child1_real = [0.0] * n
        child2_real = [0.0] * n
        for i in range(n):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta_c + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
            child1_real[i] = v1_real[i] + beta * (v2_real[i] - v1_real[i])
            child2_real[i] = v2_real[i] + beta * (v1_real[i] - v2_real[i])
            if INTEGER_ENCODING2: #min_int and max_int depend on number of filters and number of kernels.
                min_int, max_int = low_high_integer2(i)
            child1_real[i] = check_within_bounds(child1_real[i], min_int, max_int)
            child2_real[i] = check_within_bounds(child2_real[i], min_int, max_int)
        return child1_real, child2_real


    

    def __init__(self):
        self.parent1 = None
        self.parent2 = None
