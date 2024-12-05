import time, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from numpy import linalg as LA

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file
from data_cleaning import findLongestConseqSubseq

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def calculate_speed(dif_x,dif_y,ts,number_frame_scene_changing=5):
    focal_distance_fbf=np.sqrt(np.sum([dif_x**2,dif_y**2],axis=0))
    focal_distance_fbf[0:number_frame_scene_changing+1]=np.nan##plus one to include the weird data from taking difference between 0 and some value
    instant_speed=focal_distance_fbf/np.diff(ts)
    return instant_speed

def time_series_plot(target_distance,instant_speed,angles,analysis_window):
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(9, 7), tight_layout=True
    )
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 8})
    # Set the axis line width to 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['axes.linewidth'] = 2
    cmap = plt.get_cmap('viridis')
    ax1, ax2, ax3= axes.flatten()
    ax1.set(
        title='Distance'
    )
    ax2.set(
        title='Instant Speed'
    )
    ax3.set(
        title='angular deviation'
    )
    ax1.plot(np.arange(target_distance.shape[0]),target_distance)
    ax2.plot(np.arange(instant_speed.shape[0]),instant_speed)
    ax3.plot(np.arange(angles.shape[0]),angles)
    plt.show()

def trajectory_analysis(df_focal_animal):
    trajec_lim=150
    variables=np.sort(df_focal_animal[df_focal_animal['type']!='empty_trial']['mu'].unique(),axis = 0)
    fig, subplots = plt.subplots(
        nrows=1, ncols=variables.shape[0]+1, figsize=(20, 4), tight_layout=True
    )
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 8})
    # Set the axis line width to 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['axes.linewidth'] = 2
    #plt.rcParams['font.family'] = 'Helvetica'
    cmap = plt.get_cmap('viridis')
    for key, grp in df_focal_animal.groupby('fname'):
        if grp['type'][0]=='empty_trial':
            subplot_title='ISI'
            subplots[0].scatter(grp["X"].values,grp["Y"].values, c=np.arange(grp.shape[0]), marker=".", alpha=0.5)
            this_subplot=0
        else:
            for count,this_variable in enumerate(variables):
                if this_variable==grp['mu'][0]:
                    this_subplot=count+1
                    subplot_title=f'direction:{this_variable}'
                    subplots[this_subplot].scatter(grp["X"].values,grp["Y"].values, c=np.arange(grp.shape[0]), marker=".", alpha=0.5)
                else:
                    continue
        subplots[this_subplot].set(
        xlim=(-1*trajec_lim, trajec_lim),
        ylim=(-1*trajec_lim, trajec_lim),
        yticks=([-1*trajec_lim, 0, trajec_lim]),
        xticks=([-1*trajec_lim, 0, trajec_lim]),
        aspect=('equal'),
        title=subplot_title)
    plt.show()

def unwrap_degree(angle_rad,number_frame_scene_changing):
    angle_rad[np.isnan(angle_rad)] = 0
    # angle_rad=np.unwrap(angle_rad)
    # angular_velocity=np.diff(np.unwrap(angle_rad))
    ang_deg = np.mod(np.rad2deg(angle_rad),360.) ## if converting the unit to degree
    angular_velocity=np.diff(np.unwrap(ang_deg,period=360))##if converting the unit to degree
    angle_rad[0:number_frame_scene_changing+1]=np.nan ##plus one to include the weird data from taking difference between 0 and some value
    angular_velocity[0:number_frame_scene_changing+1]=np.nan  ##plus one to include the weird data from taking difference between 0 and some value   
    return angle_rad,angular_velocity

def classify_follow_epochs(focal_xy,instant_speed,ts,portion,analysis_methods):
    extract_follow_epoches=analysis_methods.get("extract_follow_epoches",True)
    analysis_window=analysis_methods.get("analysis_window")
    follow_within_distance=analysis_methods.get("follow_within_distance",50)
    focal_distance_fbf=instant_speed*np.diff(ts)
    agent_distance_fbf=np.sqrt(np.sum([np.diff(portion)[0]**2,np.diff(portion)[1]**2],axis=0))
    vector_dif=portion-focal_xy
    target_distance=LA.norm(vector_dif, axis=0)
    dot_product=np.diag(np.matmul(np.transpose(np.diff(focal_xy)),np.diff(portion)))
    angles = np.arccos(dot_product/focal_distance_fbf/agent_distance_fbf)
    angles_in_degree= angles*180/np.pi
    # if analysis_methods.get("plotting_trajectory"):
    #     time_series_plot(target_distance,instant_speed,angles_in_degree,analysis_window)
    follow_sercan=np.logical_and(target_distance[1:]<follow_within_distance, instant_speed>1,angles_in_degree<10)
    if extract_follow_epoches:
        epochs_of_interest=follow_sercan
    else:
        epochs_of_interest=np.ones((instant_speed.shape[0]))==1.0#created a all-true array for overall heatmap
    return epochs_of_interest,vector_dif
def align_agent_moving_direction(vector_dif,grp):
    theta = np.radians(grp['mu'].values[0]-360)  
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])# calculate the rotation matrix to align the agent to move along the same direction
    vector_dif_rotated=rot_matrix @ vector_dif
    return vector_dif_rotated
    #vector_dif_rotated=vector_dif_rotated[:,1:]
def conclude_as_pd(df_focal_animal,vector_dif_rotated,epochs_of_interest,fname,agent_no=0):
    num_frames=df_focal_animal[df_focal_animal["fname"]==fname].shape[0]
    degree_in_the_trial=np.repeat(df_focal_animal[df_focal_animal["fname"]==fname]["mu"].to_numpy()[0],num_frames)
    degree_time=np.vstack((degree_in_the_trial,df_focal_animal[df_focal_animal["fname"]==fname]["ts"].to_numpy()))
    degree_time=degree_time[:,1:]
    vector_dif_rotated=vector_dif_rotated[:,1:]
    follow_wrap=np.concat((vector_dif_rotated[:,epochs_of_interest],degree_time[:,epochs_of_interest]))
    follow_pd=pd.DataFrame(np.transpose(follow_wrap))
    follow_pd.insert(0, 'agent_id',np.repeat(agent_no,follow_pd.shape[0]))
    return follow_pd
def preprocess_data_for_visual_evoked_behaviour(summary_file,focal_animal_file,agent_file,analysis_methods):
    trajec_lim=150
    good_follower_only=False
    duration_for_baseline=3
    analysis_window=analysis_methods.get("analysis_window")
    monitor_fps=analysis_methods.get("monitor_fps")
    align_with_isi_onset=analysis_methods.get("align_with_isi_onset",False)
    # df_agent_list=[]
    # with h5py.File(agent_file, "r") as f:
    #     for hdf_key in f.keys():
    #         tmp_agent = pd.read_hdf(agent_file,key=hdf_key)
    #         tmp_agent.insert(0, 'type',np.repeat(hdf_key,tmp_agent.shape[0]))
    #         df_agent_list.append(tmp_agent)
    # df_agent=pd.concat(df_agent_list)
    df_focal_animal = pd.read_hdf(focal_animal_file)
    df_summary=pd.read_hdf(summary_file)
    # df_focal_animal['this_vr']=this_vr
    # df_focal_animal['fname']=df_focal_animal['fname'].astype(str) + '_' + df_focal_animal['this_vr'].astype(str)
    test = np.where(df_focal_animal["heading"].values == 0)[0]
    num_unfilled_gap=findLongestConseqSubseq(test,test.shape[0])
    dif_across_trials=[]
    if 'basedline_v' in locals():
        del basedline_v
    trial_id=0
    for key, grp in df_summary.groupby('fname'):
        focal_xy=np.vstack((df_focal_animal[df_focal_animal["fname"]==key]["X"].to_numpy(),df_focal_animal[df_focal_animal["fname"]==key]["Y"].to_numpy()))
        distance_from_centre=np.sqrt(np.sum([focal_xy[0]**2,focal_xy[1]**2],axis=0))
        ts=df_focal_animal[df_focal_animal["fname"]==key]["ts"].to_numpy()
        dif_x=np.diff(focal_xy[0])
        dif_y=np.diff(focal_xy[1])
        instant_speed=calculate_speed(dif_x,dif_y,ts)
        angle_rad = df_focal_animal[df_focal_animal["fname"]==key]["heading"].to_numpy()
        _,angular_speed=unwrap_degree(angle_rad,num_unfilled_gap)
        if 'type' in df_summary.columns:
            if align_with_isi_onset:
                if grp['type'][0]=='empty_trial':
                    frame_range=analysis_window[1]*monitor_fps
                    d_of_interest=distance_from_centre[:frame_range]
                    v_of_interest=instant_speed[:frame_range]
                    w_of_interest=angular_speed[:frame_range]
                else:
                    frame_range=analysis_window[0]*monitor_fps
                    d_of_interest=distance_from_centre[frame_range:]
                    v_of_interest=instant_speed[frame_range:]
                    w_of_interest=angular_speed[frame_range:]
            else:
                if grp['type'][0]=='empty_trial':
                    print('ISI now')
                    frame_range=analysis_window[0]*monitor_fps
                    d_of_interest=distance_from_centre[frame_range:]
                    v_of_interest=instant_speed[frame_range:]
                    w_of_interest=angular_speed[frame_range:]
                    basedline_v=np.mean(v_of_interest[-duration_for_baseline*monitor_fps:])
                    normalised_v=np.repeat(np.nan,v_of_interest.shape[0])
                    basedline_w=np.mean(w_of_interest[-duration_for_baseline*monitor_fps:])
                    normalised_w=np.repeat(np.nan,w_of_interest.shape[0])
                else:
                    print('stim now')
                    frame_range=analysis_window[1]*monitor_fps
                    d_of_interest=distance_from_centre[:frame_range]
                    v_of_interest=instant_speed[:frame_range]
                    w_of_interest=angular_speed[:frame_range]
                    if 'basedline_v' in locals():
                        normalised_v=v_of_interest/basedline_v
                    else:
                        normalised_v=np.repeat(np.nan,v_of_interest.shape[0])
                    if 'basedline_w' in locals():
                        normalised_w=w_of_interest/basedline_w
                    else:
                        normalised_w=np.repeat(np.nan,w_of_interest.shape[0])

        else:
            if align_with_isi_onset:
                if df_focal_animal[df_focal_animal["fname"]==key]['density'][0]==0.0:
                    frame_range=analysis_window[1]*monitor_fps
                    d_of_interest=distance_from_centre[:frame_range]
                    v_of_interest=instant_speed[:frame_range]
                    w_of_interest=angular_speed[:frame_range]
                    if 'basedline_v' in locals():
                        normalised_v=v_of_interest/basedline_v
                    else:
                        normalised_v=np.repeat(np.nan,v_of_interest.shape[0])
                    if 'basedline_w' in locals():
                        normalised_w=w_of_interest/basedline_w
                    else:
                        normalised_w=np.repeat(np.nan,w_of_interest.shape[0])
                else:
                    frame_range=analysis_window[0]*monitor_fps
                    d_of_interest=distance_from_centre[frame_range:]
                    v_of_interest=instant_speed[frame_range:]
                    w_of_interest=angular_speed[frame_range:]
                    basedline_v=np.mean(v_of_interest[-duration_for_baseline*monitor_fps:])
                    normalised_v=np.repeat(np.nan,v_of_interest.shape[0])
                    basedline_w=np.mean(w_of_interest[-duration_for_baseline*monitor_fps:])
                    normalised_w=np.repeat(np.nan,w_of_interest.shape[0])

            else:
                if df_focal_animal[df_focal_animal["fname"]==key]['density'][0]==0.0:
                    print('ISI now')
                    frame_range=analysis_window[0]*monitor_fps
                    d_of_interest=distance_from_centre[frame_range:]
                    v_of_interest=instant_speed[frame_range:]
                    w_of_interest=angular_speed[frame_range:]

                else:
                    print('Stim now')
                    frame_range=analysis_window[1]*monitor_fps
                    d_of_interest=distance_from_centre[:frame_range]
                    v_of_interest=instant_speed[:frame_range]
                    w_of_interest=angular_speed[:frame_range]


        if 'type' in df_summary.columns:
            con_matrex=(d_of_interest,v_of_interest,w_of_interest,normalised_v,normalised_w,np.repeat(trial_id,v_of_interest.shape[0]),np.repeat(grp['mu'][0],v_of_interest.shape[0]),np.repeat(grp['type'][0],v_of_interest.shape[0]))
        else:
            con_matrex=(d_of_interest,v_of_interest,w_of_interest,normalised_v,normalised_w,np.repeat(trial_id,v_of_interest.shape[0]),np.repeat(df_focal_animal[df_focal_animal["fname"]==key]['mu'][0],v_of_interest.shape[0]),np.repeat(df_focal_animal[df_focal_animal["fname"]==key]['density'][0],v_of_interest.shape[0]))
        raw_data=np.vstack(con_matrex)
        dif_across_trials.append(pd.DataFrame(np.transpose(raw_data)))
        trial_id += 1
    tmp=pd.concat(dif_across_trials)
    if 'type' in df_summary.columns:
        tmp.columns = ['distance_from_centre', 'velocity','omega','normalised_v','normalised_omega','id','mu','object']
    else:
        tmp.columns = ['distance_from_centre', 'velocity','omega','normalised_v','normalised_omega','id','mu','density']
    # tmp.insert(0, 'animal_id', np.repeat(animal_id,tmp.shape[0]))
    # dif_across_animals.append(tmp)
    return tmp,num_unfilled_gap

    
def calculate_relative_position(summary_file,focal_animal_file,agent_file,analysis_methods):
    pre_stim_ISI=60
    trajec_lim=150
    df_agent_list=[]
    with h5py.File(agent_file, "r") as f:
        for hdf_key in f.keys():
            tmp_agent = pd.read_hdf(agent_file,key=hdf_key)
            tmp_agent.insert(0, 'type',np.repeat(hdf_key,tmp_agent.shape[0]))
            df_agent_list.append(tmp_agent)
    df_agent=pd.concat(df_agent_list)
    df_focal_animal = pd.read_hdf(focal_animal_file)
    df_summary=pd.read_hdf(summary_file)
    test = np.where(df_focal_animal["heading"].values == 0)[0]
    num_unfilled_gap=findLongestConseqSubseq(test,test.shape[0])
    print(f"the length :{num_unfilled_gap} of unfilled gap in {focal_animal_file}")
    # if analysis_methods.get("plotting_trajectory"):
    #     trajectory_analysis(df_focal_animal)
    dif_across_trials=[]
    trial_evaluation_list=[]
    trial_id=0
    for key, grp in df_summary.groupby('fname'):
        focal_xy=np.vstack((df_focal_animal[df_focal_animal["fname"]==key]["X"].to_numpy(),df_focal_animal[df_focal_animal["fname"]==key]["Y"].to_numpy()))
        dif_x=np.diff(focal_xy[0])
        dif_y=np.diff(focal_xy[1])
        ts=df_focal_animal[df_focal_animal["fname"]==key]["ts"].to_numpy()
        instant_speed=calculate_speed(dif_x,dif_y,ts)
        heading_direction = df_focal_animal[df_focal_animal["fname"]==key]["heading"].to_numpy()
        if grp['type'][0]=='empty_trial':
            focal_distance_ISI=instant_speed*np.diff(ts)
            _,turn_degree_ISI=unwrap_degree(heading_direction,num_unfilled_gap)
            pre_stim_ISI=grp['duration'][0]
            continue
        else:
            focal_distance_fbf=instant_speed*np.diff(ts)
            agent_xy=np.vstack((df_agent[df_agent['fname']==key]["X"].to_numpy(),df_agent[df_agent['fname']==key]["Y"].to_numpy()))
            if np.isnan(np.min(agent_xy))==True:
                ##remove nan from agent's xy with interpolation
                tmp_arr=agent_xy[0]
                tmp_arr1=agent_xy[1]
                nans, x= nan_helper(tmp_arr)
                tmp_arr[nans]= np.interp(x(nans), x(~nans), tmp_arr[~nans])
                nans, y= nan_helper(tmp_arr1)
                tmp_arr1[nans]= np.interp(y(nans), y(~nans), tmp_arr1[~nans])
            if agent_xy.shape[1]>focal_xy.shape[1]:
                num_portion=round(agent_xy.shape[1]/focal_xy.shape[1])
                midpoint = agent_xy.shape[1] // num_portion
                # Loop through the array in two portions
                follow_pd_list=[]
                for i in range(num_portion):
                    if i == 0:
                        portion = agent_xy[:,:midpoint]  # First half
                        print(f"Processing first half: {portion}")
                    else:
                        portion = agent_xy[:,midpoint:]  # Second half
                        print(f"Processing second half: {portion}")
                    epochs_of_interest,vector_dif=classify_follow_epochs(focal_xy,instant_speed,ts,portion,analysis_methods)
                    vector_dif_rotated=align_agent_moving_direction(vector_dif,grp)
                    follow_pd=conclude_as_pd(df_focal_animal,vector_dif_rotated,epochs_of_interest,key,i)
                    follow_pd.insert(0, 'type',np.repeat(df_agent[df_agent['fname']==key]["type"].values[0],follow_pd.shape[0]))
                    follow_pd_list.append(follow_pd)
            else:
                epochs_of_interest,vector_dif=classify_follow_epochs(focal_xy,instant_speed,ts,agent_xy,analysis_methods)
                if grp['mu'].values[0]==45:
                    print("test")
                vector_dif_rotated=align_agent_moving_direction(vector_dif,grp)
                follow_pd=conclude_as_pd(df_focal_animal,vector_dif_rotated,epochs_of_interest,key)
                follow_pd.insert(0, 'type',np.repeat(df_agent[df_agent['fname']==key]["type"].values[0],follow_pd.shape[0]))
            
            if 'follow_pd_list' in locals():
                follow_pd_combined=pd.concat(follow_pd_list)
                dif_across_trials.append(follow_pd_combined)
                sum_follow_epochs=follow_pd_combined.shape[0]
            else:
                dif_across_trials.append(follow_pd)
                sum_follow_epochs=follow_pd.shape[0]
            _,turn_degree_fbf=unwrap_degree(heading_direction,num_unfilled_gap)
            angular_velocity=turn_degree_fbf/np.diff(ts)
            df_summary = pd.DataFrame(
                {
                    "trial_id": [trial_id],
                    "mu": [grp['mu'].values[0]],
                    "polar_angle": [grp['polar_angle'].values[0]],
                    #"this_vr": [grp['this_vr'][0]],
                    "num_follow_epochs": [sum_follow_epochs],
                    "number_frames": [focal_xy.shape[1]-1],
                    "travel_distance": [np.nansum(focal_distance_fbf)],
                    "turning_distance": [np.nansum(abs(turn_degree_fbf))],
                    "travel_distance_ISI": [np.nansum(focal_distance_ISI)],
                    "turning_distance_ISI": [np.nansum(abs(turn_degree_ISI))],
                    "duration": [grp['duration'].values[0]],
                    "duration_ISI": [pre_stim_ISI],
                    "temperature": [df_focal_animal[df_focal_animal["fname"]==key]['temperature'].values[0]],
                    "humidity": [df_focal_animal[df_focal_animal["fname"]==key]['humidity'].values[0]],
                    "object":[grp['type'].values[0]]
                }
            )
            trial_evaluation_list.append(df_summary)
            trial_id=trial_id+1
    return dif_across_trials,trial_evaluation_list,num_unfilled_gap

def load_data(this_dir, json_file):
    relative_pos_all_animals=[]
    trial_evaluation_across_animals=[]
    animal_id=0
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())

    agent_pattern = f"VR2*agent_full.h5"
    agent_file = find_file(Path(this_dir), agent_pattern)
    xy_pattern = f"VR2*XY_full.h5"
    focal_animal_file = find_file(Path(this_dir), xy_pattern)
    summary_pattern = f"VR2*score_full.h5"
    summary_file = find_file(Path(this_dir), summary_pattern)
   
    #dif_across_trials_list,trial_evaluation_list,num_unfilled_gap=calculate_relative_position(summary_file,focal_animal_file,agent_file,analysis_methods)
    _,_=preprocess_data_for_visual_evoked_behaviour(summary_file,focal_animal_file,agent_file,analysis_methods)
# #methods to load hdf file save from other format
# filename="D:/MatrexVR_2024_Data/RunData/20241116_155210/VR1_2024-11-16_155242_XY_full_test.h5"
# Timestamp = {}
# XY={}
# Heading={}
# with h5py.File(filename, "r") as f:
#     for key in f.keys():
#         #print(key)

#         ds_arr = f[key]["TimeStamp"][:] # returns as a numpy array
#         Timestamp[key] = ds_arr # appends the array in the dict under the key
#         ds_arr = f[key]["XY1"][:] # returns as a numpy array
#         XY[key]=ds_arr
#         ds_arr = f[key]["Heading"][:] # returns as a numpy array
#         Heading[key] =ds_arr

# #df = pd.DataFrame.from_dict(dictionary)
    
#    return dif_across_trials_list,trial_evaluation_list,num_unfilled_gap
if __name__ == "__main__":
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241125_131510"
    thisDir = r"D:/MatrexVR_2024_Data/RunData/20241201_131605"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")