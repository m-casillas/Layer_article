import pandas as pd

df = pd.read_csv('merged.csv')
print(df.columns)

print(f'FLOPS MIN {df["FLOPs"].min():.2e}')
print(f'FLOPS MAX {df["FLOPs"].max():.2e}')

print()
print(f'PARAMS MIN {df["Num_Params"].min():.2e}')
print(f'PARAMS MAX {df["Num_Params"].max():.2e}')
