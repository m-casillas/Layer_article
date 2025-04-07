import pandas as pd
import os

def split_all_csvs_in_folder(input_folder, path_results):
    print('Splitting CSVs per crossover and mutation type...')
    
    for input_file in os.listdir(input_folder):
        input_file_name = f'{input_file[:-4]}'
        if input_file.endswith('.csv'):
            full_input_path = os.path.join(input_folder, input_file)
            df = pd.read_csv(full_input_path)
            print(f'Spliiting {full_input_path}')
            search_types = df['Search strategy'].unique()
            cross_types = df['Crossover type'].unique()
            mut_types = df['Mutation type'].unique()
            dirs = os.path.join(path_results, input_file_name)
            os.makedirs(dirs, exist_ok=True)
            for search in search_types:
                for cross in cross_types:
                    for mut in mut_types:
                        subset = df[(df['Search strategy'] == search) & (df['Crossover type'] == cross) & (df['Mutation type'] == mut)]
                        if subset.empty:
                            continue
                        filename = f"{cross}_{mut}.csv"
                        
                        filepath = os.path.join(path_results, input_file_name, filename)
                        
                        subset.to_csv(filepath, index=False)
                        print(f"Saved {filepath}")
    
    print('Done.')

def merge_csv_files(input_dir='output', output_file='merged.csv'):
    ...
        
path =  os.getcwd()  #'/content/drive/MyDrive/DCC/Research/First_article'
csv_dir = 'csv_files'
splittedCSV_dir = 'splitted_csv'
splitCsv_dir = os.path.join(csv_dir, splittedCSV_dir)
path_splitted = os.path.join(path, splitCsv_dir)
print(f'{path_splitted = }')
filename = 'archs_2025-04-06_15-32_SPC.csv'
filepath = os.path.join(path, csv_dir)
print(f'{filepath = }')
split_all_csvs_in_folder(filepath, path_splitted)
