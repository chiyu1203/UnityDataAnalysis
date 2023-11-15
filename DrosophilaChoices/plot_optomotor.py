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
    fig.add_trace(go.Scatter(x=time_index, y=animal_data['GameObjectRotX'], mode='markers', name='Animal Rotation'))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Step (60/second)',
        yaxis_title='Rotation (degrees)',
        legend_title='Legend',
        template='plotly_dark'
    )
    
    fig.show()

#timestamp = '132700'
timestamp = '132003'
#timestamp = '121505'
#timestamp = '120613'
#timestamp = '115424'
#timestamp = '102557'
folder_path = f'/home/insectvr/src/build/20231115_optomotor_Data/RunData/20231115_{timestamp}'
grating_file = f'20231115_{timestamp}_Optomotor_Grating Generator_.csv'
grating_data = load_data(os.path.join(folder_path, grating_file))

for i in range(1, 5):
    animal_file = f'20231115_{timestamp}_Optomotor_VR{i}_.csv'
    animal_data = load_data(os.path.join(folder_path, animal_file))
    create_plot(animal_data, grating_data, f'Optomotor Response - VR{i}')
