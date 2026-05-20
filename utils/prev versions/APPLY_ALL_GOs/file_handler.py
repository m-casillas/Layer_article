from globalsENAS import *
import shutil

def split_all_csvs_in_folder(input_folder, path_results):
    print('\nSplitting CSVs per crossover and mutation type...')
    
    for input_file in os.listdir(input_folder):
        input_file_name = f'{input_file[:-4]}'
        if input_file.endswith('.csv'):
            full_input_path = os.path.join(input_folder, input_file)
            df = pd.read_csv(full_input_path)
            print(f'Spliiting {full_input_path}')
            
            if df['HHSE'].any():
                print('HHSE detected. Copying same file and renaming it to splitted folder')
                src = os.path.join(input_folder, input_file)
                dst = os.path.join(path_results, input_file)
                shutil.copyfile(src, dst)
                new_path = os.path.join(path_results, f'HHSE{"_NSGA2" if df["NSGA2"].any() else ""}_{df["HHSE_TYPE"].unique()[0]}.csv')
                os.rename(dst, new_path)
                print('Done copying HHSE file.')
                continue
            
            search_types = df['Search_strategy'].unique()
            cross_types  = df['Crossover_type'].unique()
            mut_types    = df['Mutation_type'].unique()
            #dirs = os.path.join(path_results, input_file_name)
            #os.makedirs(dirs, exist_ok=True)
            #for search in search_types:
            for cross in cross_types:
                for mut in mut_types:
                    for isNSGA2 in [True, False]:
                        #subset = df[(df['Search_strategy'] == search) & (df['Crossover_type'] == cross) & (df['Mutation_type'] == mut) & (df['NSGA2'] == isNSGA2)]
                        subset = df[(df['Crossover_type'] == cross) & (df['Mutation_type'] == mut) & (df['NSGA2'] == isNSGA2)]
                        
                        if subset.empty:
                            continue
                        filename = f"{cross}_{mut}"
                        filename += f"_NSGA2.csv" if isNSGA2 else ".csv"
                        custom_folder = ''
                        custom_dir = os.path.join(path_results, custom_folder)
                        #ensure_folder_exists(custom_dir)
                        filepath = os.path.join(custom_dir, filename)
                        subset.to_csv(filepath, index=False)
                        print(f"Saved {filepath}")

    print('Done.')

def merge_csv_files(folder_path, results_path, output_file='merged.csv', delete_duplicates=False):
    print(f'Merging CSV files in {folder_path}...')
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    output_path = os.path.join(results_path, output_file)
    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        if file == 'merged.csv':
            if len(csv_files) == 1:
                print('Only merged.csv found, skipping merge.')
                return output_path
            else:
                continue

        print(f'Reading {file}...')
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    # Remove duplicates if requested
    if delete_duplicates:
        before = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['Integer_encoding'])
        after = len(merged_df)
        print(f'Removed {before - after} duplicate rows based on Integer_Encoding.')
    # Save the merged DataFrame to a new CSV
    merged_df.to_csv(output_path, index=False)
    print(f'Merged CSV saved to {output_path}')
    return output_path

def split(experiment_folder):
    input_folder = experiment_folder
    output_folder = os.path.join(experiment_folder, 'splitted')
    ensure_folder_exists(input_folder)
    ensure_folder_exists(output_folder)
    split_all_csvs_in_folder(input_folder, output_folder)

def merge(experiment_folder):
    input_folder = experiment_folder
    #output_folder = os.path.join(experiment_folder, 'merged')
    output_folder = os.path.join(experiment_folder)
    ensure_folder_exists(input_folder)
    ensure_folder_exists(output_folder)
    merge_csv_files(input_folder, output_folder, output_file='merged.csv', delete_duplicates=True)
