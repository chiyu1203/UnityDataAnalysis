import numpy as np
import pandas as pd

def add_run_id(df):
    df['run_id'] = (df['CurrentStep'] != df['CurrentStep'].shift()).cumsum()
    return df
#add unique run id by combining run_id and trial_timestamp and vr
def add_unique_run_id(df):
    df['unique_run_id'] = df['run_id'].astype(str) + '_' + df['trial_timestamp'] + '_' + df['VR'].astype(str)
    return df


def transform_positions(df):
    """
    Transforms the SensPosX and SensPosY positions for each run_id in the DataFrame.
    The transformation is based on the position and rotation offsets calculated from row number 10.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the trajectory data with 'SensPosX', 'SensPosY', 'GameObjectRotY', 'SensRotY', and 'run_id' columns.
    
    Returns:
    pd.DataFrame: DataFrame with new columns 'TransformedPosX' and 'TransformedPosY'.
    """
    transformed_dfs = []
    
    unique_run_ids = df['run_id'].unique()
    
    for run_id in unique_run_ids:
        run_df = df[df['run_id'] == run_id].copy()
        
        if len(run_df) > 10:
            x_offset = run_df.iloc[10]['SensPosX']
            y_offset = run_df.iloc[10]['SensPosY']
            rotation_offset = run_df.iloc[10]['GameObjectRotY'] - run_df.iloc[10]['SensRotY']
            
            # Apply the offsets
            run_df['OffsetPosX'] = run_df['SensPosX'] - x_offset
            run_df['OffsetPosY'] = run_df['SensPosY'] - y_offset

            # Convert rotation offset to radians
            rotation_offset_rad = np.deg2rad(rotation_offset)
            
            # Calculate the transformed positions
            run_df['TransformedPosX'] = (
                run_df['OffsetPosX'] * np.cos(rotation_offset_rad) - 
                run_df['OffsetPosY'] * np.sin(rotation_offset_rad)
            )
            run_df['TransformedPosY'] = (
                run_df['OffsetPosX'] * np.sin(rotation_offset_rad) + 
                run_df['OffsetPosY'] * np.cos(rotation_offset_rad)
            )
            
            transformed_dfs.append(run_df)
    
    # Concatenate all transformed DataFrames
    df = pd.concat(transformed_dfs)
    return df

def discretize_space(df, space_disc_threshold):
    df['space_disc'] = False
    ref_pos = df.iloc[0]
    df.loc[0, 'space_disc'] = True
    for i, row in df.iterrows():
        if i == 0:
            continue
        xdist = ref_pos['SensPosX'] - row['SensPosX']
        ydist = ref_pos['SensPosY'] - row['SensPosY']
        xydist = np.sqrt(xdist**2 + ydist**2)
        if xydist >= space_disc_threshold:
            ref_pos = row
            df.loc[i, 'space_disc'] = True
    return df[df['space_disc']]

def calculate_direction_of_movement(df):
    df['delta_x'] = df['SensPosX'].diff()
    df['delta_y'] = df['SensPosY'].diff()
    df['movement_direction'] = np.arctan2(df['delta_y'], df['delta_x']) * (180 / np.pi)
    return df

def filter_by_mean_speed(dfs, speed_threshold=20):
    """
    Filters out runs where the mean speed is higher than the given threshold.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames containing the trajectory data with 'unique_run_id' and 'speed_mm_s' columns.
    speed_threshold (float): The speed threshold for filtering (in mm/s).

    Returns:
    list of pd.DataFrame: List of DataFrames with filtered runs.
    """
    filtered_dfs = []
    
    for df in dfs:
        unique_run_ids = df['unique_run_id'].unique()
        
        for unique_run_id in unique_run_ids:
            run_df = df[df['unique_run_id'] == unique_run_id]
            if run_df['speed_mm_s'].mean() <= speed_threshold:
                filtered_dfs.append(run_df)
    
    return filtered_dfs

def filter_by_total_displacement(dfs, distance_threshold=5):
    """
    Filters out runs where the mean speed is higher than the given threshold.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames containing the trajectory data with 'unique_run_id' and 'speed_mm_s' columns.
    speed_threshold (float): The speed threshold for filtering (in mm/s).

    Returns:
    list of pd.DataFrame: List of DataFrames with filtered runs.
    """
    filtered_dfs = []
    
    for df in dfs:
        unique_run_ids = df['unique_run_id'].unique()
        
        for unique_run_id in unique_run_ids:
            run_df = df[df['unique_run_id'] == unique_run_id]
            last_position = run_df.iloc[-1][['TransformedPosX', 'TransformedPosY']]
            total_distance = np.sqrt(last_position['TransformedPosX']**2 + last_position['TransformedPosY']**2)
            if total_distance >= distance_threshold:
                filtered_dfs.append(run_df)
    
    return filtered_dfs