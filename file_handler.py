from globalsENAS import *
import shutil
from pathlib import Path
import re

def organize_csv2(folder_path,cross_family=False,mut_family=False,only_window=False):
    """
    Organize CSV files from the 'splitted' directory.

    Expected filename patterns:
        CROSS_MUT.csv
        CROSS_MUT_*.csv
        CROSS_MUT_W##_*.csv

    Only the first token (CROSS) and second token (MUT)
    are used for family grouping.

    Window grouping is based on tokens matching W##
    where ## is one or more digits.

    Original files remain in 'splitted'.
    """

    splitted_dir=Path(folder_path)/"splitted"

    if not splitted_dir.exists():
        raise FileNotFoundError(f"'splitted' folder not found: {splitted_dir}")

    window_pattern=re.compile(r"^W\d+$")

    for csv_file in splitted_dir.glob("*.csv"):
        parts=csv_file.stem.split("_")

        if len(parts)<2:
            continue

        cross=parts[0]
        mut=parts[1]

        # Create cross-based family structure
        if cross_family:
            cross_dir=splitted_dir/"CROSS_FAMILY"/cross/"splitted"
            cross_dir.mkdir(parents=True,exist_ok=True)

            shutil.copy2(csv_file,cross_dir/csv_file.name)

        # Create mutation-based family structure
        if mut_family:
            mut_dir=splitted_dir/"MUT_FAMILY"/mut/"splitted"
            mut_dir.mkdir(parents=True,exist_ok=True)

            shutil.copy2(csv_file,mut_dir/csv_file.name)

        # Create window-based structure
        if only_window:
            window=None

            for token in parts:
                if window_pattern.match(token):
                    window=token
                    break

            if window is not None:
                window_dir=splitted_dir/window/"splitted"
                window_dir.mkdir(parents=True,exist_ok=True)

                shutil.copy2(csv_file,window_dir/csv_file.name)

def organize_csv(folder_path, cross_family=False, mut_family=False, only_window=False):
    """
    Organize CSV files from the 'splitted' directory.

    Expected filename patterns:
        CROSS_MUT.csv
        CROSS_MUT_*.csv
        CROSS_MUT_W##_*.csv

    Only the first token (CROSS) and second token (MUT)
    are used for family grouping.

    Window grouping is based on tokens matching W##
    where ## is one or more digits.

    Original files remain in 'splitted'.
    """

    splitted_dir = Path(folder_path) / "splitted"

    if not splitted_dir.exists():
        raise FileNotFoundError(f"'splitted' folder not found: {splitted_dir}")

    window_pattern = re.compile(r"^W\d+$")

    for csv_file in splitted_dir.glob("*.csv"):
        parts = csv_file.stem.split("_")

        if len(parts) < 2:
            continue

        cross = parts[0]
        mut = parts[1]

        # Combined cross + mutation family structure
        if cross_family and mut_family:
            family_name = f"{cross}_{mut}"
            family_dir = splitted_dir / family_name / "splitted"
            family_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(csv_file, family_dir / csv_file.name)

        # Cross-only family structure
        elif cross_family:
            cross_dir = splitted_dir / "CROSS_FAMILY" / cross / "splitted"
            cross_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(csv_file, cross_dir / csv_file.name)

        # Mutation-only family structure
        elif mut_family:
            mut_dir = splitted_dir / "MUT_FAMILY" / mut / "splitted"
            mut_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(csv_file, mut_dir / csv_file.name)

        # Window-based structure
        if only_window:
            window = None

            for token in parts:
                if window_pattern.match(token):
                    window = token
                    break

            if window is not None:
                window_dir = splitted_dir / window / "splitted"
                window_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(csv_file, window_dir / csv_file.name)

def split_all_csvs_in_folder(input_folder, path_results):
    print('\nSplitting CSVs per crossover and mutation type...')
    
    for input_file in os.listdir(input_folder):
        if input_file.endswith('.csv'):
            full_input_path = os.path.join(input_folder, input_file)
            df = pd.read_csv(full_input_path)
            print(f'Spliiting {full_input_path}')
            
            for search_type in df['search_name'].unique():
                subset = df[df['search_name'] == search_type]
                filename = search_type
                filename += '_generation_status' if 'generation_status' in input_file else ''
                filename += '.csv'
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
