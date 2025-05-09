import pandas as pd
import os
from globalsENAS import ensure_folder_exists

def split_all_csvs_in_folder(input_folder, path_results):
    print('\nSplitting CSVs per crossover and mutation type...')
    
    for input_file in os.listdir(input_folder):
        input_file_name = f'{input_file[:-4]}'
        if input_file.endswith('.csv'):
            full_input_path = os.path.join(input_folder, input_file)
            df = pd.read_csv(full_input_path)
            print(f'Spliiting {full_input_path}')
            search_types = df['Search_strategy'].unique()
            cross_types = df['Crossover_type'].unique()
            mut_types = df['Mutation_type'].unique()
            #dirs = os.path.join(path_results, input_file_name)
            #os.makedirs(dirs, exist_ok=True)
            for search in search_types:
                for cross in cross_types:
                    for mut in mut_types:
                        subset = df[(df['Search_strategy'] == search) & (df['Crossover_type'] == cross) & (df['Mutation_type'] == mut)]
                        if subset.empty:
                            continue
                        filename = f"{cross}_{mut}.csv"
                        #custom_folder = input_file_name + '_splitted' #Uncomment this if you want to create a folder for each input file
                        custom_folder = ''
                        custom_dir = os.path.join(path_results, custom_folder)
                        #ensure_folder_exists(custom_dir)
                        filepath = os.path.join(custom_dir, filename)
                        subset.to_csv(filepath, index=False)
                        print(f"Saved {filepath}")
    
    print('Done.')

def merge_csv_files(folder_path, results_path, output_file='merged.csv'):
    print(f'Merging CSV files in {folder_path}...')
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        print(f'Reading {file}...')
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged DataFrame to a new CSV
    output_path = os.path.join(results_path, output_file)
    merged_df.to_csv(output_path, index=False)
    print(f'Merged CSV saved to {output_path}')
    return output_path


#split()
#merge()