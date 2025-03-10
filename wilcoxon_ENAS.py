import pandas as pd
from scipy.stats import wilcoxon

def Wilcoxon_ENAS(data1, data2):

    data1 = pd.read_csv('file1.csv')
    data2 = pd.read_csv('file2.csv')

    data1_values = data1['column_name'] 
    data2_values = data2['column_name']


    stat, p_value = wilcoxon(data1_values, data2_values)

    print(f"Wilcoxon test statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        return "Not rejected"
    else:
        return "Rejected"
