import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def create_rotation_plot(animal_data, grating_data, title):
    time_index = np.array(range(len(grating_data)))  # Convert range to numpy array
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=grating_data['GameObjectRotY'], mode='markers', name='Grating Rotation'))
    fig.add_trace(go.Scatter(x=time_index, y=animal_data['GameObjectRotY'], mode='markers', name='Animal Rotation'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Step (60/second)',
        yaxis_title='Rotation (degrees)',
        legend_title='Legend',
        template='plotly_dark'
    )
    
    fig.show()

def create_sensor_position_plot(animal_data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=animal_data['SensPosX'], y=animal_data['SensPosY'], mode='markers', marker=dict(color=np.arange(len(animal_data)), colorbar=dict(title='Time Step'), colorscale='Viridis'), name='Sensor Position'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Sensor X Position',
        yaxis_title='Sensor Y Position',
        legend_title='Legend',
        template='plotly_dark'
    )
    
    fig.show()

# List of timestamps
timestamps = ['20231219_143605']

# Base folder path
base_folder_path = '/home/insectvr/src/build/20231213_7minOptomotor_Data/RunData/'

for timestamp in timestamps:
    folder_path = os.path.join(base_folder_path, timestamp)
    grating_file = f'{timestamp}_Optomotor_Grating Generator_.csv'
    grating_data = load_data(os.path.join(folder_path, grating_file))

    for i in range(1, 5):
        animal_file = f'{timestamp}_Optomotor_VR{i}_.csv'
        animal_data = load_data(os.path.join(folder_path, animal_file))
        
        # Create rotation plot
        create_rotation_plot(animal_data, grating_data, f'Optomotor Response - VR{i}')
        
        # Create sensor position plot
        create_sensor_position_plot(animal_data, f'Sensor Positions - VR{i}')
