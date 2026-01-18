#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

#IDEA: for the HHSE keep track of convergency time .
#URGENT: compare correlation between fully trained architectures and predicted ones.
#Some FLOPs can be used for previous architectures. There is a dictionary that i never used for that.
#Implement the regular NSGA-II. The one used for selecting operators is dumb.
#Defnie a method for loading the initial population from a file.
#Final idea: from the correlation matrix, choose the appropiate genetic operator
#STATS and PLOTS for RANK 1 ARCHS 
#TECNASGUI: filter option, add 'eliminate duplicates' button. Focus on the integer or binary encoding
#Instead of print everything, create a debug print. Check dprint in globalsENAS
# #LAYER REP: LAST LAYER MUST BE A FIXED POOL (?)
#ADD SWAP MUTATION TO LAYERS AND BLOCKS. MAYBE NOT TO LAYERS. BLOCKS IS DONE
#SOMEDAY: RANDOM NUM OF ARCHS IS WRONG
#NOW: A LOT OF DUPLICATED BEST ARCHS. CALCULATE CM FOR ONE AND COPY TO THE OTHERS.
#URGENT: CONSEQUTIVE POOLS ARE HAPPENING
#DEL OBJECTS WITH DESTRUCTOR (DELTE AL RREFERENCES)
#PLOTS IN SAME FIG, INSTEAD OF SEVERAL.
#ADD COMBINATION TO CORRELATION MATRICES. DELETE MEASURES THAT DO NOT APPEAR (FOR EXAMPLE NONE WITH HDs)
##CHECK VALUES FOR SUMMARIES. SUCCMUT AND SUCCCROSS ARE MISSING.
#ARCH RANDOM NAMES FOR CROSSOVER AND MUTATION AS A FUNCTION
#REPORT FOR GAs PERFORMANCE: Right now it takes the last of each generation and sum them. Next time, its only the last number per EXECUTION.
#LIMIT NUMBER OF POOLING LAYERS
#Instead of training and saving individually, do it in batches.
#If the program crashes, try to recover the last generation or execution from the csv file. This means I have to save in the excel the parents after each generation.

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



tecNAS = TECNAS(HHSE = config_tecnas.HHSE, representation_type = ConfigClass.REPRESENTATION_TYPE, regressor_type = ConfigClass.REGRESSOR_TYPE, experiment_folder = ConfigClass.experiment_folder)
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
print(f"{elapsed_minutes:.1f}min.")
print(f"{elapsed_hours:.1f} hrs.")

