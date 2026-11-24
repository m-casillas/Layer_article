#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

#BALANCING: HYPERMUTATION, NORMALMUTATION, UNDERMUTATION. Too much exploration, too much exploitation. 
#URGENT NOW: RWD with acc, flops and params not working. WTF is RWD???
#F URGENT! Plot only certain GOs. For example, for accuracy, SBX shouldnt be plotted.
#URGENT SOMEDAY: SBX crossover it's being called with max_int as 63 (because of blocks) This has to be changed for Layers. Also for mutation. 
#Determine stage order as parameter (CROSS, MUT, HV, GD)  or another one. I changed HHSE_TEC
#HHSE_TEC: Add name to search type in CSVs. Also to folders. HHSE_TEC COMBINED and with fixed criteria, like CROSS. Ablation studies.
#NOW: Generation Status plotter should be given the generation intervals.
#UREGNT: GReedy COMBINED should return 4 markov matrices. First 20% generations give the CROSS Matrix, etc. Analyze those matrices.
#GREEDY: How to break ties? CHECK HOW IT WAS DONE FOR TEC_AOS.
#NOW: A LOT OF DUPLICATED BEST ARCHS. CALCULATE CM FOR ONE AND COPY TO THE OTHERS.
#Defnie a method for loading the initial population from a file.
#STATS and PLOTS for RANK 1 ARCHS 
#TECNASGUI: filter option, add 'eliminate duplicates' button. Focus on the integer or binary encoding
#ADD SWAP MUTATION TO LAYERS AND BLOCKS. MAYBE NOT TO LAYERS. BLOCKS IS DONE
#SOMEDAY: RANDOM NUM OF ARCHS IS WRONG

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

import sys
import ast
import os
import time
from globalsENAS import *

if len(sys.argv) > 1:
    print('Running with arguments')
    #python main.py REAL UC,TPC,SPC MSWAP,None False
    arguments = sys.argv[1:]
    args_encoding = arguments[0]
    args_cross_list = arguments[1].split(',')
    args_mut_list = arguments[2].split(',')
    args_INT2 = ast.literal_eval(arguments[3])
    args_hyper_mut = ast.literal_eval(arguments[4])
    ConfigClass.ENCODING_TYPE = args_encoding
    config_tecnas.ENCODING_TYPE = args_encoding
    ConfigClass.crossover_types_list = args_cross_list
    config_tecnas.crossover_types_list = args_cross_list
    ConfigClass.mutation_types_list = args_mut_list
    config_tecnas.mutation_types_list = args_mut_list
    config_tecnas.INTEGER_ENCODING2 = args_INT2
    config_tecnas.HYPER_MUTATION = args_hyper_mut
    print("ARGUMENTOS RECIBIDOS:", *arguments, flush=True)
    
from Genotype import *
from Architecture import *
from LayerRepresentation import *
from Mutator import *
from Crossover import *
from TECNAS import *
from PlotterENAS import *
from ReportENAS import *

os.system("cls")
os.system("clear")

#ConfigClass.mutation_types_list = ['LOL']
tecNAS = TECNAS(HHSE = config_tecnas.HHSE, NSGA2 = config_tecnas.NSGA2, representation_type = ConfigClass.REPRESENTATION_TYPE, encoding_type = ConfigClass.ENCODING_TYPE, regressor_type = ConfigClass.REGRESSOR_TYPE, experiment_folder = ConfigClass.experiment_folder)
start_time = time.time()
tecNAS.mainLoop()

end_time = time.time()
# Calculate elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600.
print()
print(Fore.LIGHTBLUE_EX + f"TECNAS DONE!!!!!!!!!!!!!!")
print(f"{elapsed_seconds:.1f} sec.")
print(f"{elapsed_minutes:.1f} min.")
print(f"{elapsed_hours:.1f} hrs.")

