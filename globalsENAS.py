import os #
path =  os.getcwd()  #'/content/drive/MyDrive/DCC/Research/First_article'
dir_results = 'final_project_results'
fig_results = 'final_project_figures'
path_results = os.path.join(path, dir_results)
path_figures = os.path.join(path, fig_results)

layer_types = ['INP', 'CONV', 'POOLMAX', 'FLATTEN', 'DENSE']
layer_mutation_types = ['L_MODIFY_PARAMS', 'L_CHANGE_TYPE']
layer_search_strategy = ['RANDOM', 'GA']
MINMAX_LAYERS = [3,20]
CONV_KERNEL_LIST = [3,5,7]
POOL_KERN_LIST = [2,3]
NUM_FILTERS_LIST = [64, 128, 256, 512]
DENSE_NEURONS_LIST = [16, 32, 64, 128, 256]

#Each parameter is encoded as an integer for the genetic operators
CONV_KERNELS = {i:value for i,value in enumerate(CONV_KERNEL_LIST)}
POOL_KERNELS = {i:value for i,value in enumerate(POOL_KERN_LIST)}
#NUM_FILTERS = {0:64, 1:128, 2:256, 3:512}
NUM_FILTERS = {i:value for i,value in enumerate(NUM_FILTERS_LIST)}
DENSE_NEURONS = {i:value for i,value in enumerate(DENSE_NEURONS_LIST)}
ACTIVATION_FUNCTIONS = {0:'relu', 1:'sigmoid', 2:'tanh', 3:'softmax'}

