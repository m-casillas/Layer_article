#Cuts accuracy history to epochs

import pandas as pd
import ast

df = pd.read_csv('BLOCKS_INTEGERS_BEST360_TECNAS2_HD.csv', encoding='utf-8')
epochs = 70

def truncate_history(row):
    acc_hist = ast.literal_eval(row['Accuracy_history'])   # convert string → list
    acc_hist70 = acc_hist[:epochs]                         # truncate list

    row['Epochs'] = epochs
    row['Accuracy_history'] = str(acc_hist70)
    row['Accuracy'] = acc_hist70[-1] if acc_hist70 else None

    return row

df = df.apply(truncate_history, axis=1)

df.to_csv('lol.csv', index=False)
print("Done.")