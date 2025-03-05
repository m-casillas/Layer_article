from globalsENAS import *
"""#Utilities

"""

def find_median(pop_list):
    sorted_array = sorted(pop_list, key=lambda arch:arch.acc)
    n = len(sorted_array)
    median_arch = sorted_array[n // 2]  # Odd length, middle element
    idx = [i for i, arch in enumerate(sorted_array) if arch.acc == median_arch.acc]
    return median_arch, idx

def calculate_model_flops(model):
    # Ensure the model is built, required for input shape
    if not model.built:
        model.build(input_shape=(None,) + model.input_shape[1:])

    # Define the forward pass for the model
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    # TensorFlow Profiler gets FLOPs
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

    flops = graph_info.total_float_ops // 2 # //2 `profile` counts multiply and accumulate as two flops
    return flops

def create_pool_layer(ks = None):
        #A Pool layer has: kernel size (ks)... more to come
        #If ks is None, it is randomly created
        if ks == None:
            ks = random.choice(POOL_KERN)
        return {'POOLMAX':ks}

def create_conv_layer(nf = None, ks = None):
        #A Conv layer has: Number of filters (nf) kernel size (ks)... more to come
        #If nf and ks are None, it is randomly created
        if nf == None and ks == None:
            nf = random.randint(NUM_FILTERS[0], NUM_FILTERS[1]+1)
            ks = random.choice(CONV_KERN)
        return {'CONV':[nf, ks]}

def create_dense_layer(nn = None, act = None):
        #A Dense layer has: Number of neurons (nn) and activation function (act)... more to come
        #If nn and act are None, it is randomly created
        if nn == None and act == None:
            nn = random.randint(DENSE_NEURONS[0], DENSE_NEURONS[1]+1)
        return {'DENSE':[nn, 'relu']}

def create_last_name(column_list):
    last_name = ''
    for c in column_list:
            last_name = last_name + c + '_'
    return last_name

def determine_label_filename(filename):
    if 'L_MODIFY_PARAMS' in filename:
        label = 'L_MODIFY_PARAMS'
    elif 'L_CHANGE_TYPE' in filename:
        label = 'L_CHANGE_TYPE'
    elif 'NONE' in filename:
        label = 'RANDOM'
    else:
        label = '?'
    return label