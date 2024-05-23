import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_trajectories_and_polar_histograms(dfs, timestamp, directory_path):
    for i, df in enumerate(dfs):
        df['source'] = f'df{i+1}'
    df_combined = pd.concat(dfs)
    steps = df_combined['run_id'].unique()
    num_bins = 36

    for step_index, step in enumerate(steps):
        filtered_df = df_combined[df_combined['CurrentStep'] == step]
        sources = filtered_df['source'].unique()
        fig, axes = plt.subplots(3, len(sources), figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

        if len(sources) == 1:
            axes = np.array([axes])

        for i, source in enumerate(sources):
            try:
                source_df = filtered_df[filtered_df['source'] == source]
                ax_traj = axes[0][i]
                sc = ax_traj.scatter(source_df['SensPosX'], source_df['SensPosY'], c=np.arange(len(source_df)), cmap='viridis')
                ax_traj.set_title(f'{source} - CurrentStep {step}')
                ax_traj.set_xlabel('SensPosX')
                ax_traj.set_ylabel('SensPosY')
                ax_traj.set_aspect('equal', adjustable='box')

                ax_polar1 = fig.add_subplot(3, len(sources), len(sources) + i + 1, polar=True)
                direction_values = source_df['movement_direction'] / 180 * np.pi
                counts, bin_edges = np.histogram(direction_values, bins=num_bins, range=(-np.pi, np.pi))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax_polar1.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='k')

                ax_polar2 = fig.add_subplot(3, len(sources), 2*len(sources) + i + 1, polar=True)
                rot_x_values = source_df[source_df['GameObjectRotY'] != 0]['GameObjectRotY']
                counts, bin_edges = np.histogram(rot_x_values, bins=num_bins, range=(0, 2*np.pi))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax_polar2.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='k')
                ax_polar2.set_theta_zero_location('N')
                ax_polar2.set_theta_direction(-1)

            except Exception as e:
                print(f"An error occurred while plotting for source {source}: {e}")

        plt.tight_layout()
        figure_name = f'{directory_path}{timestamp}_trajectories_and_histograms_step_{step_index}.png'
        plt.savefig(figure_name)
        print(f"Saved: {figure_name}")

def plot_trajectory(df, title="Trajectory Plot"):
    """
    Plots the trajectory of an animal based on SensPosX and SensPosY.

    Parameters:
    df (pd.DataFrame): DataFrame containing the trajectory data with 'SensPosX' and 'SensPosY' columns.
    title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['SensPosX'], df['SensPosY'], marker='o', linestyle='-', markersize=2)
    plt.title(title)
    plt.xlabel('SensPosX')
    plt.ylabel('SensPosY')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

def plot_trajectories_by_run_id(df, title_prefix="Trajectory Plot"):
    """
    Plots the trajectory of an animal for each run identified by 'run_id'.

    Parameters:
    df (pd.DataFrame): DataFrame containing the trajectory data with 'SensPosX', 'SensPosY', and 'run_id' columns.
    title_prefix (str): Prefix for the plot titles.
    """
    unique_run_ids = df['run_id'].unique()
    
    for run_id in unique_run_ids:
        run_df = df[df['run_id'] == run_id]
        plt.figure(figsize=(10, 6))
        plt.plot(run_df['SensPosX'], run_df['SensPosY'], marker='o', linestyle='-', markersize=2)
        plt.title(f"{title_prefix} - Run ID {run_id}")
        plt.xlabel('SensPosX')
        plt.ylabel('SensPosY')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()

def plot_polar_histograms_by_run_id(df, title_prefix="Polar Histogram"):
    """
    Plots the polar histogram of GameObjectRotY for each run identified by 'run_id'.

    Parameters:
    df (pd.DataFrame): DataFrame containing the trajectory data with 'GameObjectRotY' and 'run_id' columns.
    title_prefix (str): Prefix for the plot titles.
    """
    unique_run_ids = df['run_id'].unique()
    num_bins = 36

    for run_id in unique_run_ids:
        run_df = df[df['run_id'] == run_id]
        rot_y_values = run_df[run_df['GameObjectRotY'] != 0]['GameObjectRotY']
        if rot_y_values.empty:
            continue

        plt.figure(figsize=(8, 8))
        direction_values = rot_y_values / 180 * np.pi
        counts, bin_edges = np.histogram(direction_values, bins=num_bins, range=(0, 2*np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.subplot(projection='polar')
        plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='k')
        plt.title(f"{title_prefix} - Run ID {run_id} - Step {run_df['CurrentStep'].iloc[0]}")
        plt.show()
