def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_running_in_colab():
    import os
    from google.colab import drive
    drive.mount('/content/drive')
    get_ipython().system(f"pip install colorama")
    get_ipython().system(f"pip install -U scikit-learn==1.2.0")
    get_ipython().system(f"pip install pympler")

    os.chdir('/content/drive/MyDrive/DCC/Research/Layer_article')

from TECNAS import *
from globalsENAS import *
import random
from config_tecnas import set_FLAGS

class Trainer():

    def get_new_intvec_from_file(self, file = ''):
        #Loads a csv file with architectures and returns a list of integer vectors
        path_nofile = self.training_folder
        listdir = os.listdir(path_nofile)

        for filename in listdir:
            if filename.endswith(".csv"):
                df_full_path = os.path.join(path_nofile, filename)
                print(f'Loading architectures from the file {df_full_path}')
                df = pd.read_csv(df_full_path)
                for i in df.index.tolist():
                    layer_idxs = []
                    arch_idx = df.loc[i, 'ID']
                    gen_list = df.loc[i, 'Genotype']
                    gen_list = ast.literal_eval(gen_list)
                    for layer in gen_list:
                        params, = layer.values() #Get the [32,2] for example.
                        if get_key_from_value(layer, params) == 'INP':
                            continue
                        if get_key_from_value(layer, params) == 'FLATTEN':                        
                            continue
                        if get_key_from_value(layer, params) == 'LAST_DENSE':
                            continue
                        layer_idx = Globals.all_layers.index(layer)
                        #print(f'{layer = } {layer_idx = }')
                        layer_idxs.append(layer_idx)
                    df.loc[i, ['Integer_encoding']] = [str(layer_idxs)]
            df.to_csv(df_full_path, index=False)
        print(f'Integer encodings added to {df_full_path}')

    def train_from_file(self, file = ''):
            '''
            Trains all architectures from the file in the training arena folder.
            It creates a file called merged_filtered.csv, this has to change
            It takes one architecture, trains it, and adds it to trained_results.csv
            Before training, checks if the architecture was already trained from the CURRENT.CSV file. (Pending)
            '''
            path_nofile = self.training_folder
            listdir = os.listdir(path_nofile)
            df_current = pd.read_csv(os.path.join(path_nofile, 'CURRENT.CSV')) if 'CURRENT.csv' in listdir else None
            for filename in listdir:
                #Ignore the training results or the filtered csv.
                if '_training_' in filename:
                    continue

                if filename.endswith(".csv"):
                    df_full_path = os.path.join(path_nofile, filename)
                    file_results = os.path.join(path_nofile, f'{filename[:-4]}_training_results.csv')
                    file_filtered = os.path.join(path_nofile, f'{filename[:-4]}_training_filtered.csv')

                    if any("_filtered" in word for word in listdir):
                        print(f'Loading architectures from the file {file_filtered}')
                        
                    else:
                        print(f'{file_filtered} not found. Creating file')
                        df = pd.read_csv(df_full_path)
                        df = df.drop_duplicates(subset='Integer_encoding', keep='first')
                        df.to_csv(file_filtered, index=False)
                        print(f'{file_filtered} created')

                    df_filtered = pd.read_csv(file_filtered)
                    #At this point, some archs may have Trained_Completely as True. This gets the last update. This df is not saved as csv
                    df_filtered_temp = df_filtered[df_filtered['Trained_Completely'] == False]

                    if any("_training_results" in word for word in listdir) == False: #If the training_results file does not exist, create it
                        df_trained = pd.DataFrame(columns=df_filtered.columns)
                        df_trained.to_csv(file_results, index=False)

                    df_trained = pd.read_csv(file_results)
                    num_archs = len(df_filtered_temp)
                    print(f'Found {num_archs} architectures to train in {file_filtered}')
                    
                    if num_archs == 0:
                        return
                    #Take each row, train it and save it. use indexes. k
                    cont = 0
                    for i in df_filtered_temp.index.tolist():
                        cont += 1
                        print(Fore.LIGHTBLUE_EX + f'{cont}/{num_archs} Fully Training architecture {df_filtered_temp.loc[i, "ID"]} {i = }' + Style.RESET_ALL)
                        arch_idx = df_filtered_temp.loc[i, 'ID']
                        gen_list = df_filtered_temp.loc[i, 'Genotype']
                        gen_list = ast.literal_eval(gen_list)
                        arch = self.tecnas.create_individual(gen_list = gen_list)
                        arch.idx = arch_idx
                        arch.ID = df_filtered_temp.loc[i, 'ID']

                        dfTraining_index = len(df_trained)
                        df_trained.loc[dfTraining_index] = df_filtered.loc[i]
                        arch = self.tecnas.train_model(arch,  print_status = False)
                        
                        df_trained.loc[i, ['Epochs', 'Accuracy', 'Accuracy_history', 'Loss_history','Trained_Completely']] = [ConfigClass.EPOCHS, arch.acc, str(arch.acc_hist), str(arch.loss_hist), True]
                        print(f'Updating info for {arch.ID} at {i = } in the file {file}')
                        df_filtered.loc[i, 'Trained_Completely'] = True #Only merged_filtered.csv changes this value
                        df_filtered.to_csv(file_filtered, index=False)
                        print(f'Saving info in {file_results}\n')
                        df_trained.loc[[dfTraining_index]].to_csv(file_results, mode='a', header=False, index=False)
                    

    def calculate_confusion_matrix_metrics_from_file(self, getflops = True, get_numparams = True):
            #Compute confusion matrix from BEST architectures from LAST GENERATION for each execution.
            path_nofile = self.training_folder
            listdir = os.listdir(path_nofile)

            for filename in listdir:
                #Ignore the cm results or the filtered csv.
                if '_cm_' in filename:
                    continue

                if filename.endswith(".csv"):
                    df_full_path = os.path.join(path_nofile, filename)
                    file_cm = os.path.join(path_nofile, f'{filename[:-4]}_cm_results.csv')
                    file_filtered = os.path.join(path_nofile, f'{filename[:-4]}_cm_filtered.csv')
                    
                    if any("_filtered" in word for word in listdir):
                        print(f'Loading architectures from the file {file_filtered}')
                       
                    else:
                        print(f'{file_filtered} not found. Creating file')
                        df = pd.read_csv(df_full_path)
                        df = df[df['Has_CM'] == False]
                        df.to_csv(file_filtered, index=False)
                        print(f'{file_filtered} created')

                    df_filtered = pd.read_csv(file_filtered)
                    #At this point, some archs may have Has_CM as True. This gets the last update. This df is not saved as csv
                    df_filtered_temp = df_filtered[df_filtered['Has_CM'] == False] 
                    
                    if any("_cm_results" in word for word in listdir) == False: #If the cm_results file does not exist, create it
                        df_cm = pd.DataFrame(columns=df_filtered.columns)
                        df_cm.to_csv(file_cm, index=False)

                    df_cm = pd.read_csv(file_cm)
                    num_archs = len(df_filtered_temp)
                    print(f'Found {num_archs} architectures')
                    if num_archs == 0:
                        return
                    #Take each row, train it and save it. use indexes. k
                    cont = 0
                    for i in df_filtered_temp.index.tolist():
                        cont += 1
                        print(Fore.LIGHTBLUE_EX + f'{cont}/{num_archs} Calculating CM for {df_filtered.loc[i, "ID"]} {i = }' + Style.RESET_ALL)
                        arch_idx = df_filtered_temp.loc[i, 'ID']
                        gen_list = df_filtered_temp.loc[i, 'Genotype']
                        gen_list = ast.literal_eval(gen_list)
                        arch = self.tecnas.create_individual(gen_list = gen_list)
                        arch.idx = arch_idx
                        arch.ID = df_filtered.loc[i, 'ID']
                                        
                        dfCM_index = len(df_cm)
                        df_cm.loc[dfCM_index] = df_filtered.loc[i]

                        self.tecnas.getflops = getflops
                        self.tecnas.get_numparams = get_numparams
                            
                        arch = self.tecnas.train_model(arch = arch, calculate_cm = True, epochs = ConfigClass.EPOCHS, print_status = False)
                        df_cm.loc[i, ['Accuracy', 'Has_CM','Confusion_matrix', 'cm_accuracy', 'cm_precision_macro', 'cm_recall_macro', 'cm_f1_macro']] = [arch.acc, True, str(arch.cm),arch.cm_accuracy,arch.cm_precision_macro,arch.cm_recall_macro, arch.cm_f1_macro]
                        df_cm.loc[i, ['Epochs', 'Accuracy_history', 'Loss_history','Trained_Completely']] = [ConfigClass.EPOCHS, str(arch.acc_hist), str(arch.loss_hist), True]
                        df_cm.loc[i, ['FLOPs', 'Num_Params']] = [arch.flops, arch.num_params]
                        df_cm.loc[i, ['Top1', 'Top5']] = [arch.top1, arch.top5]
                        #df_cm.loc[i, ['Accuracy', 'Has_CM','Confusion_matrix', 'cm_accuracy', 'cm_precision_macro', 'cm_recall_macro', 'cm_f1_macro']] = [7, 7, 7, 7, 7,7, 7]
                        #df_cm.loc[i, ['Epochs', 'Accuracy_history', 'Loss_history','Trained_Completely']] = [EPOCHS, 7, 7, 7]

                        print(f'Updating info for {arch.ID} at {i = } in the file {filename}')
                        df_filtered.loc[i, 'Has_CM'] = True #Only merged_filtered.csv changes this value 
                        df_filtered.to_csv(file_filtered, index=False)
                        print(f'Saving info in {file_cm}\n')
                        df_cm.loc[[dfCM_index]].to_csv(file_cm, mode='a', header=False, index=False)


    def compare_surrogate_to_real(self, surrogate_model_idx = 0):
        regressor_folder = os.path.join(path, 'results', 'comparing_surrogate_real', 'surrogates', 'regressors')
        self.surrogate = Surrogate_ENAS(model_choice = surrogate_model_idx, regressor_folder = regressor_folder)

        path_nofile = os.path.join(path, 'results', 'comparing_surrogate_real')

        files = [f for f in os.listdir(path_nofile) if os.path.isfile(os.path.join(path_nofile, f))]
        for file in files:
            if 'comparison_results' in file:
                continue
            path_file = os.path.join(path_nofile, file)
            path_comparison_results = os.path.join(path_nofile, f'{file[:-4]}_comparison_results.csv')
            print(f'Loading architectures from the file {path_file}')
            df = pd.read_csv(path_file)
            df_comparison = pd.DataFrame(columns=['ID', 'Genotype', 'Epochs', 'Surrogate_acc', 'Real_acc'])
            print(Fore.LIGHTBLUE_EX + f'Number or architectures to compare: {len(df)}' + Style.RESET_ALL)
            for i in df.index.tolist():
                print(Fore.LIGHTYELLOW_EX + f'Processing architecture {i+1}/{len(df)}' + Style.RESET_ALL)
                arch_idx = df.loc[i, 'ID']
                gen_list = df.loc[i, 'Genotype']
                gen_list = ast.literal_eval(gen_list)
                arch = self.tecnas.create_individual(gen_list = gen_list)
                arch.idx = arch_idx
                self.surrogate.load_arch(arch)
                arch.acc = self.surrogate.predict_arch(arch.idx, model_choice = surrogate_model_idx)
                idx = df.loc[i, 'ID']
                epochs = df.loc[i, 'Epochs']
                genotype = df.loc[i, 'Genotype']
                real_acc = df.loc[i, 'Accuracy']
                
                df_comparison.loc[i, ['ID', 'Genotype', 'Epochs', 'Surrogate_acc', 'Real_acc', 'Delta_acc']] = [idx, str(genotype), epochs, str(arch.acc), str(real_acc), abs(arch.acc - real_acc)]
    
                df_comparison.to_csv(path_comparison_results, index=False)
                print(f'Saving comparison results in {path_comparison_results}\n')
        

    def __init__(self, representation_type = 'L', experiment_folder = '', SURROGATE = 0, TRAIN = 1, SIMULATE = 0):
        #self.archStatus = archStatus
        self.training_folder = os.path.join(experiment_folder, "training_arena")
        ensure_folder_exists(self.training_folder)
        self.tecnas = TECNAS(representation_type, experiment_folder = self.training_folder, regressor_type = ConfigClass.REGRESSOR_TYPE)
        self.trained_path = os.path.join(self.training_folder, 'trained_archs')
        ensure_folder_exists(self.trained_path)
        
        #self.tecnas.path_results = os.path.join(training_path, self.trained_path)
        self.tecnas.exec = 0
        self.tecnas.generation = 0
        self.tecnas.crossover_type = ''
        self.tecnas.mutation_type = ''
        self.tecnas.arch_count = 0
        self.tecnas.search_strategy = ''
        self.tecnas.SURROGATE = SURROGATE
        self.tecnas.TRAIN = TRAIN
        self.tecnas.SIMULATE = SIMULATE
        self.tecnas.REPORT_ARCH = False
        self.tecnas.SURROGATE, self.tecnas.TRAIN, self.tecnas.SIMULATE = set_FLAGS(SURROGATE, TRAIN, SIMULATE)
        print(Fore.LIGHTBLUE_EX + f'Trainer initialized with SURROGATE = {self.tecnas.SURROGATE}, TRAIN = {self.tecnas.TRAIN}, SIMULATE = {self.tecnas.SIMULATE}' + Style.RESET_ALL)



#TODO: load from folder.
experiment_folder = os.path.join(path, 'results')
trainer = Trainer(representation_type = ConfigClass.REPRESENTATION_TYPE, experiment_folder = experiment_folder, SURROGATE = 0, TRAIN = 1, SIMULATE = 0)
#trainer.get_new_intvec_from_file()
trainer.train_from_file()
#trainer.compare_surrogate_to_real(10)
#trainer.calculate_confusion_matrix_metrics_from_file(getflops = True, get_numparams = True)
