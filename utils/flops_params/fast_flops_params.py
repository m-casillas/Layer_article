import ast
import math


def flat_str(str1):
    str1 = str1.replace('(', '').replace(')', '')
    return str1

def count_flops(arch, input_res=32):
    """
    Returns total FLOPs using analytic formulas.
    Assumes square input: input_res x input_res
    """
    arch = flat_str(arch)
    arch = ast.literal_eval(arch)
    H = W = input_res
    C_in = 3
    total_flops = 0

    for layer in arch:
        name, params = next(iter(layer.items()))
        if name == "INP":
            H = W = params
            C_in = 3

        elif name == "CONV":
            C_out, K = params
            if C_in is None:
                raise ValueError("First CONV must define input channels implicitly")

            # Conv FLOPs: 2 * H * W * C_out * (C_in * K * K)
            conv_flops = 2 * H * W * C_out * (C_in * K * K)
            total_flops += conv_flops

            # BatchNorm FLOPs: ~4 ops per element
            bn_flops = 4 * H * W * C_out
            total_flops += bn_flops

            # ReLU FLOPs: 1 op per element
            relu_flops = H * W * C_out
            total_flops += relu_flops

            C_in = C_out

        elif name == "POOLMAX":
            _, K = params
            # MaxPool FLOPs ~ K*K comparisons per output
            pool_flops = H * W * C_in * (K * K)
            total_flops += pool_flops

            H //= K
            W //= K

        elif name == "GLOBAL_AVG":
            # Sum + divide
            total_flops += H * W * C_in
            H = W = 1

        elif name == "FLATTEN":
            pass

        elif name in ("DENSE", "LAST_DENSE"):
            units, _ = params
            in_features = H * W * C_in
            dense_flops = 2 * in_features * units
            total_flops += dense_flops

            C_in = units
            H = W = 1

        else:
            raise ValueError(f"Unknown layer type: {name}")

    return total_flops


def count_flops2(arch, input_res=32, nconv_per_block=2):
    """
    Accurate FLOP counter aligned with decode() implementation
    """
    arch = ast.literal_eval(arch)

    H = W = input_res
    C_in = 3
    total_flops = 0

    # ---- First CONV (outside blocks) ----
    first_conv = arch[1]["CONV"]
    C_out, K = first_conv

    # Conv
    total_flops += 2 * H * W * C_out * (C_in * K * K)
    # BN + ReLU
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
            block_flops += 4 * H * W * C_out   # BN
            block_flops += H * W * C_out       # ReLU
            C_in = C_out

        # ---- Projection shortcut if needed ----
        if block_C_in != C_in:
            # 1x1 conv projection
            block_flops += 2 * H * W * C_in * block_C_in
            block_flops += 4 * H * W * C_in   # BN

        # ---- Residual add ----
        block_flops += H * W * C_in

        # ---- Pooling ----
        if pool:
            Kp = 2
            H_out = H // Kp
            W_out = W // Kp
            block_flops += H_out * W_out * C_in * (Kp * Kp - 1)
            H, W = H_out, W_out

        total_flops += block_flops

    # ---- Global Average Pool ----
    total_flops += H * W * C_in
    H = W = 1

    # ---- Dense ----
    units, _ = arch[-1]["LAST_DENSE"]
    total_flops += 2 * C_in * units

    return total_flops


def count_params(arch):
    """
    Returns (trainable_params, non_trainable_params)
    """
    arch = flat_str(arch)
    arch = ast.literal_eval(arch)
    C_in = 3
    H = W = None

    trainable = 0
    non_trainable = 0

    for layer in arch:
        name, params = next(iter(layer.items()))

        if name == "INP":
            H = W = params
            C_in = 3

        elif name == "CONV":
            C_out, K = params
            if C_in is None:
                raise ValueError("First CONV must define input channels implicitly")

            # Conv params (bias=False)
            conv_params = C_in * C_out * K * K
            trainable += conv_params

            # BatchNorm params
            # gamma, beta → trainable
            trainable += 2 * C_out
            # running mean, var → non-trainable
            non_trainable += 2 * C_out

            C_in = C_out

        elif name == "POOLMAX":
            _, K = params
            H //= K
            W //= K

        elif name == "GLOBAL_AVG":
            H = W = 1

        elif name == "FLATTEN":
            pass

        elif name in ("DENSE", "LAST_DENSE"):
            units, _ = params
            in_features = H * W * C_in

            # Dense params (with bias)
            trainable += in_features * units + units

            C_in = units
            H = W = 1

        else:
            raise ValueError(f"Unknown layer type: {name}")

    return trainable + non_trainable


import pandas as pd
df = pd.read_csv('file.csv')
print(df.columns)
df.dropna(subset=['FLOPs'], inplace=True)
df.to_csv('file.csv', index = False)

df['flopsFast'] = df['Genotype'].apply(count_flops)
df['paramsFast'] = df['Genotype'].apply(count_params)
df['FLOPs'] = df['FLOPs']
df['flopsD'] = df['FLOPs'] - df['flopsFast']
df['paramsD'] = df['Num_Params'] - df['paramsFast']
print(df['flopsD'].mean())
print(df['paramsD'].mean())

df = df[['ID', 'Genotype', 'FLOPs', 'flopsFast', 'flopsD', 'Num_Params', 'paramsFast', 'paramsD']]

df.to_csv('file2.csv', index = False)


#g ="[{'INP': 32}, {'CONV': [64, 3]}, ({'CONV': [32, 5]}, {'CONV': [256, 3]}), ({'CONV': [32, 5]}, {'CONV': [128, 5]}, {'POOLMAX': [-1, 2]}), ({'CONV': [128, 5]}, {'CONV': [256, 5]}), ({'CONV': [128, 3]}, {'CONV': [128, 5]}, {'POOLMAX': [-1, 2]}), ({'CONV': [256, 3]}, {'CONV': [256, 5]}), ({'CONV': [32, 3]}, {'CONV': [256, 3]}, {'POOLMAX': [-1, 2]}), ({'CONV': [256, 3]}, {'CONV': [256, 3]}), ({'CONV': [128, 3]}, {'CONV': [256, 5]}, {'POOLMAX': [-1, 2]}), {'GLOBAL_AVG': None}, {'LAST_DENSE': [100, 'softmax']}]"
#print(flat_str(g))
#p1 = count_params_fast(g)
#print(p1)

import pandas as pd
from scipy.stats import spearmanr
df = pd.read_csv('file2.csv')
rho, p_value = spearmanr(df['FLOPs'], df['flopsFast'])

print("Spearman correlation:", rho)
print("p-value:", p_value)

rho, p_value = spearmanr(df['Num_Params'], df['paramsFast'])

print("Spearman correlation:", rho)
print("p-value:", p_value)