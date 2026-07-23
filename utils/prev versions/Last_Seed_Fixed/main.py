#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

#HHSE_TEC: try different stages.
#PLOT families: Cross with fixed mutation. Mutation with fixed crossover, etc. 
#Determine stage order as parameter (CROSS, MUT, HV, GD)  or another one. I changed HHSE_TEC
#Add best_acc to generation_status
#PLOTS: add GD
#HHSE_TEC: Add name to search type in CSVs. Also to folders. HHSE_TEC COMBINED and with fixed criteria, like CROSS. Ablation studies.
#TODO: Ablation studies: analyze CROSS_NONE, MUT_NON, etc. Only keep best operators for COMBINATION *dont use all of them.
#NOW: Generation Status plotter should be given the generation intervals.
#TODO: Print the k top search strategies by AUC.
#UREGNT: GReedy COMBINED should return 4 markov matrices. First 20% generations give the CROSS Matrix, etc. Analyze those matrices.
#GREEDY: How to break ties?
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


from Genotype import *
from Architecture import *
from LayerRepresentation import *
from Mutator import *
from Crossover import *
from TECNAS import *
from PlotterENAS import *
from ReportENAS import *
from globalsENAS import *


os.system("cls")
os.system("clear")


#experiment_folder = r"experiment01_cifar10complete"
#C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\surrogate_only
#experiment_folder = r"experiment01_cifar10complete"

print("REGRESSOR_TYPE =", ConfigClass.REGRESSOR_TYPE)

tecNAS = TECNAS(HHSE = config_tecnas.HHSE, NSGA2 = config_tecnas.NSGA2, representation_type = ConfigClass.REPRESENTATION_TYPE, regressor_type = ConfigClass.REGRESSOR_TYPE, experiment_folder = ConfigClass.experiment_folder)
start_time = time.time()
#tecNAS.ENAS()
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

