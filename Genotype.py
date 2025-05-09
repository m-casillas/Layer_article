from globalsENAS import *
"""# Genotype Class"""

class Genotype:
    def __str__(self):
        return 'GENOTYPE' + str(self.gen_list)

    def __init__(self, rep_type = 'L', encoding_type = 'IV', gen_list = None):
        self.rep_type = rep_type # L, B, C, T: layer, block, cell, topological
        self.encoding_type = encoding_type #IV, BV: integer vector, binary vector
        #gen_list is a list of dictionaries. Each dictionary represents the layer type and its parameters
        #INP: input, CONV, POOLMAX, FLATTEN, DENSE
        self.gen_list = gen_list