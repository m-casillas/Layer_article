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
CONV_KERN = [3,5,7]
POOL_KERN = [2,3]
NUM_FILTERS = [5,10]#[3,256]
DENSE_NEURONS = [16,256]
