import numpy as np

def identify_tracking_errors(df, speed_threshold, rotation_threshold):
    df['speed'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    df['tracking_error'] = (df['speed'] > speed_threshold) | (df['GameObjectRotY'].diff().abs() > rotation_threshold)
    return df[df['tracking_error']]

def classify_activity(df, speed_threshold):
    df['activity'] = np.where(df['speed'] > speed_threshold, 'moving', 'standing_still')
    return df
