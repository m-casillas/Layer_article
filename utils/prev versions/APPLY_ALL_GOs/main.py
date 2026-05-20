#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]

#GREEDY FINDINGS: self.pop is not chaning (good). Dealing with the status save status after choosing the best GO. What is best_go?? Check apply_all_GOs
#URGENT!!!!!!!!!: Check if the whole GREEDY is working. Also, generation_status.csv is not correct. Chosen operator should be blank, until the selected operator is applied.
#URGENET. For plots, check if you have to show average per generation or execution. For bestarch and GOs. For GOs it works per generation, per execution. Is it correct?
#URGENT: For GREEDY, some values from the last generation are passed for the next generation (such as succ_cross and succ_mut). Check it.
#Check if self.GOs_history is correct
#Print convergence plots for DPB and HV. Also for succesful crossovers and mutations (check values at the end of the execution)
#Check WHY a specific combination gave the highest accurayc or HV (use the GREEDY approach)

#NOW: A LOT OF DUPLICATED BEST ARCHS. CALCULATE CM FOR ONE AND COPY TO THE OTHERS.
#Defnie a method for loading the initial population from a file.
#STATS and PLOTS for RANK 1 ARCHS 
#TECNASGUI: filter option, add 'eliminate duplicates' button. Focus on the integer or binary encoding
#ADD SWAP MUTATION TO LAYERS AND BLOCKS. MAYBE NOT TO LAYERS. BLOCKS IS DONE
#SOMEDAY: RANDOM NUM OF ARCHS IS WRONG
#URGENT: CONSEQUTIVE POOLS ARE HAPPENING


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

