import ast

def count_params(arch):
    """
    Returns (trainable_params, non_trainable_params)
    Aligned with decode() and residual_block()
    """
    arch = ast.literal_eval(arch)

    C_in = 3
    H = W = None

    trainable = 0
    non_trainable = 0

    # ---- First CONV (outside blocks) ----
    first_conv = arch[1]["CONV"]
    C_out, K = first_conv

    # Conv (bias=False)
    trainable += C_in * C_out * K * K

    # BatchNorm
    trainable += 2 * C_out       # gamma, beta
    non_trainable += 2 * C_out   # running mean, var

    C_in = C_out

    # ---- Residual blocks ----
    for block in arch[2:-2]:
        assert isinstance(block, tuple)

        block_C_in = C_in
        conv_layers = []

        for layer in block:
            if "CONV" in layer:
                conv_layers.append(layer["CONV"])

        # ---- Convs inside block ----
        for C_out, K in conv_layers:
            trainable += C_in * C_out * K * K
            trainable += 2 * C_out
            non_trainable += 2 * C_out
            C_in = C_out

        # ---- Projection shortcut if channels differ ----
        if block_C_in != C_in:
            # 1x1 conv projection
            trainable += block_C_in * C_in
            # BN after projection
            trainable += 2 * C_in
            non_trainable += 2 * C_in

    # ---- Global Average Pool ----
    # no parameters

    # ---- Dense ----
    units, _ = arch[-1]["LAST_DENSE"]
    trainable += C_in * units + units

    return trainable# + non_trainable

def count_flops(arch, input_res=32, nconv_per_block=2):
    """ Accurate FLOP counter aligned with decode() implementation """
    arch = ast.literal_eval(arch)
    H = W = input_res
    C_in = 3
    total_flops = 0
    # ---- First CONV (outside blocks) ----
    first_conv = arch[1]["CONV"]
    C_out, K = first_conv
    total_flops += 2 * H * W * C_out * (C_in * K * K)
    total_flops += 4 * H * W * C_out
    total_flops += H * W * C_out
    C_in = C_out
    # ---- Residual Blocks ----
    for block in arch[2:-2]:
        assert isinstance(block, tuple)
        block_C_in = C_in
        block_flops = 0
        conv_layers = []
        pool = False
        for layer in block:
            if "CONV" in layer:
                conv_layers.append(layer["CONV"])
            elif "POOLMAX" in layer:
                pool = True
        # ---- Convs inside block ----
        for i, (C_out, K) in enumerate(conv_layers):
            block_flops += 2 * H * W * C_out * (C_in * K * K)
            block_flops += 4 * H * W * C_out  # BN
            block_flops += H * W * C_out       # ReLU
            C_in = C_out
        # ---- Projection shortcut if needed ----
        if block_C_in != C_in:
            block_flops += 2 * H * W * C_in * block_C_in
            block_flops += 4 * H * W * C_in  # BN
        # ---- Residual add ----
        block_flops += H * W * C_in
        # ---- Pooling ----
        if pool:
            Kp = 2
            H_out = H // Kp
            W_out = W // Kp
            block_flops += H_out * W_out * C_in * (Kp * Kp - 1)
            H = H_out   # <--- ADD THIS LINE
            W = W_out   # <--- ADD THIS LINE
        total_flops += block_flops
    # ---- Global Average Pool ----
    total_flops += H * W * C_in
    # ---- Dense ----
    units, _ = arch[-1]["LAST_DENSE"]
    total_flops += 2 * C_in * units
    return total_flops

'''
import pandas as pd
df = pd.read_csv('file.csv')
print(df.columns)
df.dropna(subset=['FLOPs'], inplace=True)
df.to_csv('file.csv', index = False)

df['flopsFast'] = df['Genotype'].apply(count_flops)
df['paramsFast'] = df['Genotype'].apply(count_params)
df['FLOPs'] = df['FLOPs']*2
df['flopsD'] = df['FLOPs'] - df['flopsFast']
df['paramsD'] = df['Num_Params'] - df['paramsFast']
print(df['flopsD'].mean())
print(df['paramsD'].mean())

df = df[['ID', 'Genotype', 'FLOPs', 'flopsFast', 'flopsD', 'Num_Params', 'paramsFast', 'paramsD']]

df.to_csv('file2.csv', index = False)
'''
import pandas as pd
from scipy.stats import spearmanr
df = pd.read_csv('file2.csv')
rho, p_value = spearmanr(df['FLOPs'], df['flopsFast'])

print("Spearman correlation:", rho)
print("p-value:", p_value)

rho, p_value = spearmanr(df['Num_Params'], df['paramsFast'])

print("Spearman correlation:", rho)
print("p-value:", p_value)