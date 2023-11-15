# Rewriting the code to include regular expressions and other fixes

import os
from pathlib import Path
import re
from collections import defaultdict
import pandas as pd
import gzip
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def pair_files_in_folder(folder_path):
    # Initialize a defaultdict to hold the file groups
    file_groups = defaultdict(list)
    
    # Define the regular expressions for locust and environment files
    locust_re = r"VR(\d+)_\.csv\.gz"  # Made it more specific
    environment_re = r"SimulatedLocustData_SimulatedLocustsVR(\d+)_\d+_\d+_\d+_\d+_\d+_\d+-\d+-\d+_\d+-\d+-\d+\.csv\.gz"  # Made it more specific
    
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        locust_match = re.search(locust_re, filename)
        environment_match = re.search(environment_re, filename)
        
        if locust_match:
            vr_number = locust_match.groups()[0]
            file_groups[vr_number].append({'type': 'locust', 'filename': filename})
            
        if environment_match:
            vr_number = environment_match.groups()[0]
            file_groups[vr_number].append({'type': 'environment', 'filename': filename})

    return file_groups


# Function to load and clean data
def load_and_clean_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        df = pd.read_csv(f)
    # Removing the first row
    df = df.iloc[1:]
    return df

def prepare_data_from_pairs(folder_path):
    file_groups = pair_files_in_folder(folder_path)
    prepared_data = {}
    
    # Loop through the file pairs
    for key, files in file_groups.items():
        locust_file = None
        environment_file = None
        print(f"Debugging for key {key}")  # Debugging line
        
        # Identify locust and environment files
        for file_info in files:
            if file_info['type'] == 'locust':
                locust_file = file_info['filename']
            elif file_info['type'] == 'environment':
                environment_file = file_info['filename']
        
        print(f"Locust file: {locust_file}")  # Debugging line
        print(f"Environment file: {environment_file}")  # Debugging line
                
        # Only proceed if both files are present
        if locust_file and environment_file:
            # Read the CSV files into Pandas dataframes
            with gzip.open(f"{folder_path}/{locust_file}", 'rb') as f:
                locust_df = pd.read_csv(f)
            with gzip.open(f"{folder_path}/{environment_file}", 'rb') as f:
                environment_df = pd.read_csv(f)
            
            # Drop the first row from both dataframes
            locust_df = locust_df.iloc[1:]
            environment_df = environment_df.iloc[1:]
            
            # Store the prepared dataframes
            prepared_data[key] = {'locust': locust_df, 'environment': environment_df}
    
    return prepared_data

                

# Your find_closest_timestamp function
def find_closest_timestamp(real_timestamp, simulated_timestamps):
    distances = np.abs(pd.to_datetime(simulated_timestamps) - pd.to_datetime(real_timestamp))
    closest_time = simulated_timestamps.loc[distances.idxmin()]
    closest_rows = simulated_timestamps[simulated_timestamps == closest_time]
    return closest_rows.index

def animate(i, locust_df, environment_df, sc_real, sc_sim, past_x_sim, past_z_sim, trail_length, trail_opacity, ax):
    real_data = locust_df.iloc[:i*30+1]
    x_real = real_data['InsectPosZ']
    z_real = real_data['InsectPosX']
    sc_real.set_offsets(np.c_[x_real, z_real])
    
    real_timestamp = real_data['Current Time'].iloc[-1]
    closest_index = find_closest_timestamp(real_timestamp, environment_df['Timestamp'])
    sim_data = environment_df.iloc[closest_index]
    x_sim = sim_data['Z']
    z_sim = sim_data['X']
    sc_sim.set_offsets(np.c_[x_sim, z_sim])

    # Store the new positions
    past_x_sim.append(x_sim)
    past_z_sim.append(z_sim)

    # Remove the oldest positions if the length exceeds trail_length
    if len(past_x_sim) > trail_length:
        del past_x_sim[0]
        del past_z_sim[0]

    # Plot the trails with decreasing opacity
    for x, z, alpha in zip(past_x_sim, past_z_sim, trail_opacity):
        ax.scatter(x, z, c='gray', alpha=alpha)

    # Update the most recent positions
    sc_sim.set_offsets(np.c_[x_sim, z_sim])


def create_animation(locust_df, environment_df, animation_filename, trail_length = 10 ):
    # Create the plot
    fig, ax = plt.subplots()
    
    # Initialize scatter plots
    sc_real = ax.scatter([], [], c='black', alpha=0.5, label='Real')
    sc_sim = ax.scatter([], [], c='gray', alpha=0.5, label='Simulated')

    # Initialize lists to store the past positions
    past_x_sim = []
    past_z_sim = []

    # Define trail properties
    trail_length = 10  # Number of past positions to display
    trail_opacity = np.linspace(0.1, 0.5, trail_length)  # Opacity values for the trail
    
    # Set labels, title, and axis limits
    ax.set_xlabel('Z Position')
    ax.set_ylabel('X Position')
    ax.set_title('Locust Movement and Behavior')
    x_min, x_max = locust_df['InsectPosX'].min(), locust_df['InsectPosX'].max()
    z_min, z_max = locust_df['InsectPosZ'].min(), locust_df['InsectPosZ'].max()
    ax.set_xlim(min(z_min, x_min)-10, max(z_max, x_max)+10)
    ax.set_ylim(min(z_min, x_min)-10, max(z_max, x_max)+10)
    
    # Create the animation
    ani = FuncAnimation(fig, lambda i: animate(i, locust_df, environment_df, sc_real, sc_sim, past_x_sim, past_z_sim, trail_length, trail_opacity, ax), frames=(len(locust_df)//30), interval=50)
    
    # Create a writer object
    writer = FFMpegWriter(fps=20)
    
    # Save the animation
    ani.save(animation_filename, writer=writer)



def generate_all_animations_in_all_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        print(folder_name)
        if os.path.isdir(folder_path):
            prepared_data = prepare_data_from_pairs(folder_path)
            for key, data in prepared_data.items():
                animation_filename = f"{folder_path}/locust_behavior_with_environment_{key}.mp4"
                create_animation(data['locust'], data['environment'], animation_filename)

# Example usage
root_folder = "/Users/apaula/src/VRDataAnalysis/SimulatedLocustSwarm/data/20230915rundata/selected"  # Replace with the path to your RunData folder
generate_all_animations_in_all_folders(root_folder)
