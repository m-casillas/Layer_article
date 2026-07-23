from globalsENAS import *


class Architecture(ABC):

    @classmethod
    def set_arch_name(self, actual_idx):
        if ConfigClass.RANDOM_NAMES == True:
            return random.choice(ARCH_NAMES_LIST)
        else:
            return actual_idx
        
    def set_report_path(self, tecnasObj = None):
        folder_name = f'{tecnasObj.representation_type}{tecnasObj.encoding_type}{architecture_filename_noCSV}'
        ensure_folder_exists(os.path.join(tecnasObj.path_results, folder_name))
        NSGA2_str = 'NSGAII_' if config_tecnas.NSGA2 else ''
        NSGA2_WINDOW_str = 'WINDOW_' if config_tecnas.NSGA2_WINDOW else ''
        NSGA2_WINDOW_SIZE_PERC_str = f"{config_tecnas.NSGA2_WINDOW_SIZE_PERC*100:.0f}_" if config_tecnas.NSGA2_WINDOW else ''
        HHSE_str = 'HHSE_' if config_tecnas.HHSE else ''
        GREEDY_str = 'GREEDY_' if config_tecnas.HHSE_GREEDY else ''
        GREEDY_CRITERIA_str = config_tecnas.HHSE_GREEDY_CRITERIA if config_tecnas.HHSE_GREEDY else ''
        GREEDY_FIXED_str = 'GREEDY_FIXED_' if config_tecnas.HHSE_GREEDY_FIXED else ''
        search_type = NSGA2_str + NSGA2_WINDOW_str + NSGA2_WINDOW_SIZE_PERC_str + HHSE_str + GREEDY_str + GREEDY_CRITERIA_str + GREEDY_FIXED_str
        #Agrega ConfigClass.SORT_ARCHS para INFERIOR, MIDDLE, SUPERIOR
        
        
        architecture_csv_filename_extended = f'{folder_name}_E{ConfigClass.EXECUTIONS}_G{ConfigClass.GENERATIONS}_N{ConfigClass.MAIN_NPOP}_{search_type}.csv'                       
        self.path_folder = os.path.join(tecnasObj.path_results, f'{folder_name}')
        self.path_filereport = os.path.join(tecnasObj.path_results, f'{folder_name}', f'{architecture_csv_filename_extended}')

    def __str__(self):
        deci = 4
        ast = 50*'-'
        arch_info = [ast,
            f"Architecture ID: {self.idx}",
            #f"Type: {self.arch_type}",
            f"Accuracy: {self.acc:.4f}",
            f"FLOPs: {self.flops}",
            f"Number of Parameters: {self.num_params}",
            f"HDP1: {self.dP1}",
            f"HDP2: {self.dP2}",
            f"HDBeforeMutation: {self.dBM}",
            f"Integer Encoding: {self.integer_encoding}",
            f"NFHT: {self.NFHT:f}",
            ast+"\n"]
        return "\n".join(arch_info)
    
    def set_genotype(self, genotype):
        self.genotype = copy.deepcopy(genotype) #object of Genotype class
        self.genoStr = str(self.genotype.gen_list)

    def set_genoStr(self):
        self.genoStr = str(self.genotype.gen_list)
    
    def __init__(self, encoding = 'INT', idx = '9999', genotypeObj = None):
        self.encoding = encoding # S: Sequential
        self.idx = idx
        self.acc_hist = [0 for i in range(ConfigClass.EPOCHS)]
        self.loss_hist = [0 for i in range(ConfigClass.EPOCHS)]
        self.acc = 0
        self.loss = 0
        self.top1 = 0
        self.top5 = 0
        self.flops = 0
        self.cpu_hours = 0
        self.num_params = 0
        self.sizeMB = 0
        self.genotype = genotypeObj #object of Genotype class
        if is_None_or_empty(genotypeObj):
            self.genoStr = ''
        else:
            self.genoStr = str(self.genotype.gen_list)
        self.integer_encoding = None #Integer encoding of the architecture
        self.binary_encoding = [] #Binary encoding of the architecture
        self.P1Idx = ''
        self.P2Idx = ''
        self.P1IntegerEncoding = None
        self.P2IntegerEncoding = None
        self.before_mutationIdx = ''
        self.before_mutationIntegerEndcoding = None
        self.P1acc = 0
        self.P2acc = 0
        self.BMacc = 0

        self.model = None #Model built with layers, etc, using keras.
        #Hamming distance between this architecture and the parent1 architecture (between encodings)
        self.dP1 = 0
        self.dP2 = 0
        self.dBM = 0
        self.dPB = 0 #Distance of the current best arch to the previous best one.
        self.isChild = False #Use this for the architecture info report
        self.isMutant = False
        self.wasInvalid = False
        self.archStatus = '' #BEST, WORST, MEDIAN of the generation
        self.trained_epochs = ConfigClass.EPOCHS
        self.gen_creation = 0 #Generation of the architecture creation

        self.cm = None #Confusion matrix
        self.cm_accuracy = 0
        self.cm_precision_macro = 0
        self.cm_recall_macro = 0
        self.cm_f1_macro = 0
    
        self.NFHT = np.nan #Normalized First Hitting Time
        self.integer_size = 0 #Size of the integer encoding
        self.succ_cross_ratio = 0
        self.succ_mut_ratio = 0
        self.model = None

        self.path_folder = ''
        self.path_filereport = ''
        

    @abstractmethod
    def decode(self):
        pass
    