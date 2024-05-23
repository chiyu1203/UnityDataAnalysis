import pandas as pd

def load_data(file_paths):
    dataframes = [pd.read_csv(fp) for fp in file_paths]
    for df in dataframes:
        df['Current Time'] = pd.to_datetime(df['Current Time'])
    return dataframes

def load_and_prepare_data(timestamp, directory_path):
    file_paths = [directory_path + f'{timestamp}_ChoiceAssay_VR{i}_.csv' for i in range(1, 5)]
    return load_data(file_paths)