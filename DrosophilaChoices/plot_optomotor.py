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

#timestamp = '132700'
#timestamp = '132003'
#timestamp = '121505'
#timestamp = '120613'
#timestamp = '115424'
#timestamp = '102557'
#timestamp = '20231129_114803'
#timestamp = '20231129_133727'
#timestamp = '20231129_145109'
#optomotor testing:
timestamp = '20231213_125040'
timestamp = '20231213_130740'
timestamp = '20231213_140734'
timestamp = '20231213_141955'
timestamp = '20231213_142939'
timestamp = '20231213_144108'
timestamp = '20231213_145503'

#timestamp = '20231129_133727'
timestamp = '20231219_113700'

#folder_path = f'/home/insectvr/src/build/20231115_optomotor_Data/RunData/20231115_{timestamp}'
#folder_path = f'/Users/apaula/Nextcloud/locustVR/locustVR_data/RunData_20231129/{timestamp}/'
folder_path = f'/home/insectvr/src/build/20231213_7minOptomotor_Data/RunData/{timestamp}/'

grating_file = f'{timestamp}_Optomotor_Grating Generator_.csv'
grating_data = load_data(os.path.join(folder_path, grating_file))

for i in range(1, 5):
    animal_file = f'{timestamp}_Optomotor_VR{i}_.csv'
    animal_data = load_data(os.path.join(folder_path, animal_file))
    create_plot(animal_data, grating_data, f'Optomotor Response - VR{i}')
