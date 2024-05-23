import numpy as np

def add_run_id(df):
    df['run_id'] = (df['CurrentStep'] != df['CurrentStep'].shift()).cumsum()
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
