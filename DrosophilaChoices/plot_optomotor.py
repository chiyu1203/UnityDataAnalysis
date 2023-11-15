import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def create_plot(animal_data, grating_data, title):
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

folder_path = '/Users/apaula/Nextcloud/locustVR/locustVR_data/20231113_150126'
grating_file = '20231113_150126_Optomotor_GratingGenerator_.csv'
grating_data = load_data(os.path.join(folder_path, grating_file))

for i in range(1, 5):
    animal_file = f'20231113_150126_Optomotor_VR{i}_.csv'
    animal_data = load_data(os.path.join(folder_path, animal_file))
    create_plot(animal_data, grating_data, f'Optomotor Response - VR{i}')
