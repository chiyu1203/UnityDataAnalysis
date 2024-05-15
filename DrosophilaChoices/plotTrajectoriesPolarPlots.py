import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data(timestamp, directory_path):
    file_paths = [directory_path + f'{timestamp}_ChoiceAssay_VR{i}_.csv' for i in range(1, 5)]
    dataframes = [pd.read_csv(fp) for fp in file_paths]
    for df in dataframes:
        df['Current Time'] = pd.to_datetime(df['Current Time'])
    return dataframes

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
    df_copy = df.copy()
    df_copy['delta_x'] = df_copy['SensPosX'].diff()
    df_copy['delta_y'] = df_copy['SensPosY'].diff()
    df_copy['movement_direction'] = np.arctan2(df_copy['delta_y'], df_copy['delta_x']) * (180 / np.pi)
    return df_copy

def plot_trajectories_and_polar_histograms_2(df1, df2, df3, df4, timestamp, directory_path):
    df1['source'] = 'df1'
    df2['source'] = 'df2'
    df3['source'] = 'df3'
    df4['source'] = 'df4'
    df_combined = pd.concat([df1, df2, df3, df4])
    steps = df_combined['CurrentStep'].unique()
    num_bins = 36

    for step_index, step in enumerate(steps):
        filtered_df = df_combined[df_combined['CurrentStep'] == step]
        sources = filtered_df['source'].unique()
        fig, axes = plt.subplots(3, len(sources), figsize=(15, 15), 
                                 gridspec_kw={'height_ratios': [3, 1, 1]})

        if len(sources) == 1:
            axes = np.array([axes])  # Ensure axes is a 2D array even for a single source

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
                direction_values = source_df['movement_direction'] /180 * np.pi
                counts, bin_edges = np.histogram(direction_values, bins=num_bins, range=(-np.pi, np.pi))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax_polar1.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='k')
                ax_polar2 = fig.add_subplot(3, len(sources), 2*len(sources) + i + 1, polar=True)
                rot_x_values = source_df[source_df['GameObjectRotY']!= 0]['GameObjectRotY']
                counts, bin_edges = np.histogram(rot_x_values, bins=num_bins, range=(0, 2*np.pi))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax_polar2.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='k')
                ax_polar2.set_theta_zero_location('N')
                ax_polar2.set_theta_direction(-1)

            except Exception as e:
                print(f"An error occurred while plotting for source {source}: {e}")

        plt.tight_layout()
         # Save the figure before showing it
        figure_name = f'{directory_path}{timestamp}_trajectories_and_histograms_step_{step_index}.png'
        plt.savefig(figure_name)
        print(f"Saved: {figure_name}")

        #plt.show()

def main():
    #timestamp = '20231129_140109'
    #timestamp = '20231129_114803'
    #timestamp = '20231129_133727'
    timestamp = '20231122_121845'
    timestamp = '20231122_122641'
    timestamp = '20231122_131112'
    timestamp = '20231122_133530'
    timestamp = '20231122_161757'
    timestamp = '20231220_160646'
    timestamp = '20240514_162026'

    #directory_path = '/Users/apaula/Nextcloud/locustVR/locustVR_data/RunData_20231129/' + timestamp + '/'
    directory_path = '/Users/apaula/Nextcloud/locustVR/locustVR_data/20231122_RunData/' + timestamp + '/'
    directory_path = '/home/insectvr/src/build/20231220_stripefixation_Data/RunData/' + timestamp + '/'
    directory_path = '/home/insectvr/src/build/20240514_testVR_Data/RunData/' + timestamp + '/'
    df1, df2, df3, df4 = load_and_prepare_data(timestamp, directory_path)
    print("Dataframe loaded:", df1.head())
    space_disc_threshold = 0.1  # Define the threshold for space discretization
    dfs_discretized = [discretize_space(df, space_disc_threshold) for df in [df1, df2, df3, df4]]
    dfs_discretized = [calculate_direction_of_movement(df) for df in dfs_discretized]

    plot_trajectories_and_polar_histograms_2(*dfs_discretized, timestamp, directory_path)

if __name__ == '__main__':
    main()