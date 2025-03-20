import os 

def list_to_dictionary(list1):
     return {i:value for i,value in enumerate(list1)}
     

path =  os.getcwd()  #'/content/drive/MyDrive/DCC/Research/First_article'
dir_results = 'final_project_results'
fig_results = 'final_project_figures'
path_results = os.path.join(path, dir_results)
path_figures = os.path.join(path, fig_results)

layer_types = ['INP', 'CONV', 'POOLMAX', 'FLATTEN', 'DENSE']
layer_mutation_types = ['L_MODIFY_PARAMS', 'L_CHANGE_TYPE']
layer_search_strategy = ['RANDOM', 'GA']
INPUT_SIZE = 32
MINMAX_LAYERS = [3,20]
CONV_KERNEL_LIST = [3,5,7]
POOL_KERN_LIST = [2,3]
NUM_FILTERS_LIST = [64, 128, 256, 512]
DENSE_NEURONS_LIST = [10, 16, 32, 64, 128, 256]
ACTIVATION_FUNCTIONS_LIST = ['relu', 'sigmoid', 'tanh', 'softmax']
LAYERS_TYPES_LIST = ['CONV', 'POOLMAX', 'DENSE']

#Each parameter is encoded as an integer for the genetic operators
CONV_KERNELS = list_to_dictionary(CONV_KERNEL_LIST)
POOL_KERNELS = list_to_dictionary(POOL_KERN_LIST)
#NUM_FILTERS = {0:64, 1:128, 2:256, 3:512}
NUM_FILTERS = list_to_dictionary (NUM_FILTERS_LIST)
DENSE_NEURONS = list_to_dictionary(DENSE_NEURONS_LIST)
ACTIVATION_FUNCTIONS = list_to_dictionary(ACTIVATION_FUNCTIONS_LIST)
LAYERS_TYPES = list_to_dictionary(LAYERS_TYPES_LIST)

#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':[-1,2]}, {'CONV':[64,3]}, {'POOLMAX':[-1,2]}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
