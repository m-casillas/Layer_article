import os
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
from globalsENAS import ensure_folder_exists
from itertools import combinations

def Wilcoxon_ENAS(experiment_folder):
    splitted_folder = os.path.join(experiment_folder, 'splitted')
    files = os.listdir(splitted_folder)
    pairs_of_files = list(combinations(files,2))
    report_columns = ['File1', 'File2', 'Wilcoxon Statistic', 'P-value', 'Result']
    df_report = pd.DataFrame(columns=report_columns)

    for file1, file2 in pairs_of_files:

        print(f"Wilcoxon test statistic: {file1} vs {file2}")
        file1_path = os.path.join(splitted_folder, file1)
        file2_path = os.path.join(splitted_folder, file2)
        # Read the CSV files
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        df2 = pd.read_csv(file2_path, encoding='utf-8')

        df1Best = df1[df1['BestGen'] == True]
        df2Best = df2[df2['BestGen'] == True]

        data1 = df1Best['Accuracy']
        data2 = df2Best['Accuracy']

        stat, p_value = wilcoxon(data1, data2)

        filename1 = file1[:-4]
        filename2 = file2[:-4]
        
        if p_value < 0.05:
            result = "Not rejected"
        else:
            result = "Rejected"
        df_report.loc[len(df_report)] = [filename1, filename2, stat, p_value, result]
    report_folder = os.path.join(experiment_folder, 'stats','Wilcoxon')
    ensure_folder_exists(report_folder)
    df_report.to_csv(os.path.join(report_folder, f'Wilcoxon_report.csv'), index=False)
    print(f"Wilcoxon test statistic: {file1} vs {file2} complete\n")
        
    


def correlation_matrix(filepath):
    df = pd.read_csv(filepath)  # Replace with your actual file path
    df = df[df['BestGen'] == True]
    # Select only columns A, B, and E
    print(df.columns)

    selected_cols = df[['Accuracy', 'FLOPs', 'Num_Params']]

    # Compute the correlation matrix
    corr_matrix = selected_cols.corr()

    # Plot the heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"Correlation Matrix")
    plt.show()

os.system("cls")
path = r"C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\experiment2"
filename = 'TPC_MPAR.csv'
filepath = os.path.join(path, filename)
#correlation_matrix(filepath)
Wilcoxon_ENAS(path)