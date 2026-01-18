import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('merged.csv')

print(df.shape)

plt.hist(df['Accuracy'])
plt.show()

plt.boxplot(df['Accuracy'])
plt.show()

print()
print(df['ID'].duplicated().sum())
print(df['Integer_encoding'].duplicated().sum())

duplicated_mask = df['Integer_encoding'].duplicated(keep=False)
repeated_indices = df.index[duplicated_mask].tolist()
print("Indices of rows with duplicated 'A' values:", repeated_indices)

df2 = df.drop_duplicates(subset=['Integer_encoding'], keep='first')
print(df2['Integer_encoding'].duplicated().sum())
