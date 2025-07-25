import time, sys, json,warnings
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linalg as LA
from scipy.stats import circmean
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file
from data_cleaning import findLongestConseqSubseq,interp_fill
colormap_name = "viridis"
sm = cm.ScalarMappable(cmap=colormap_name)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
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

def sort_raster_fictrac(raster_across_animals_fictrac,animal_interest,step_interest,analysis_methods,all_evaluation,var1,var2=None):
    analysis_window=analysis_methods.get("analysis_window")
    split_stationary_moving_ISI=analysis_methods.get("split_stationary_moving_ISI")
    monitor_fps=analysis_methods.get("monitor_fps")
    walk_threshold=1
    raster_interest=[]
    column_list = ["step_id", "elapsed_time", "instant_speed", "instant_angular_velocity"]
    n_datapoints=(analysis_window[1]-analysis_window[0])*monitor_fps
    for this_animal in animal_interest:
        this_raster=raster_across_animals_fictrac[this_animal]
        this_evaluation=all_evaluation[all_evaluation['animal_id']==this_animal]
        X=this_raster["X"].to_numpy()
        Y=this_raster["Y"].to_numpy()
        step_id=this_raster["step_id"].to_numpy()
        elapsed_time=this_raster["elapsed_time"].to_numpy()
        rot_y=this_raster["heading_angle"].to_numpy()
        rot_y = interp_fill(rot_y)
        _, turn_degree_fbf = diff_angular_degree(rot_y,0,False)
        instant_angular_velocity = turn_degree_fbf /np.diff(elapsed_time)
        instant_speed = calculate_speed(np.diff(X),np.diff(Y),elapsed_time,0)
        #plt.hist(instant_speed)
        degree_time = np.vstack(
        (
            step_id[:-1],
            elapsed_time[:-1],
            instant_speed,
            instant_angular_velocity
        ))
        follow_pd = pd.DataFrame(np.transpose(degree_time))
        follow_pd.columns=column_list
        convert_types = {'step_id':int}
        follow_pd = follow_pd.astype(convert_types)
        for this_step in step_interest:    
            analysis_ts=follow_pd[follow_pd['step_id']==this_step]['elapsed_time'].values[0]+analysis_window[0]
            condition_index = follow_pd.index[follow_pd['elapsed_time']>analysis_ts].tolist()
            start_idx = condition_index[0]
            following_rows = follow_pd.iloc[start_idx:start_idx + n_datapoints]
            this_v=following_rows["instant_speed"].to_numpy()
            #this_w=following_rows["instant_angular_velocity"].to_numpy()
            pd_to_extract=follow_pd[follow_pd['elapsed_time']>analysis_ts][:n_datapoints]
            heading_angle=rot_y[start_idx:start_idx + n_datapoints]
            heading_angle_0=heading_angle-heading_angle[abs(analysis_window[0]*monitor_fps)]
            heading_angle_0=np.unwrap(heading_angle_0, period=360)
            if split_stationary_moving_ISI:
                run_trial=np.mean(this_v[:abs(analysis_window[0]*monitor_fps)])>walk_threshold
                pd_to_extract.insert(0,'run_trial',np.repeat(run_trial,n_datapoints))
            pd_to_extract.insert(0,'heading',heading_angle_0)
            pd_to_extract.insert(0, 'frame_count', np.arange(n_datapoints))
            if 'step_id' in this_evaluation.columns:
                pd_to_extract.insert(0, var1, np.repeat(this_evaluation[this_evaluation['step_id']==int(this_step)][var1].to_numpy(),n_datapoints))
            # elif 'trial_id' in this_evaluation.columns:
            #     pd_to_extract.insert(0, var1, np.repeat(this_evaluation[this_evaluation['trial_id']==int((this_step-1)/2)][var1].to_numpy(),n_datapoints))
            # if var2 != None:
            #     pd_to_extract.insert(0, var2, np.repeat(this_evaluation[this_evaluation['trial_id']==int((this_step-1)/2)][var2].to_numpy(),n_datapoints))
            pd_to_extract.insert(0,'animal_id',np.repeat(this_animal,n_datapoints))
            #pd_to_extract = pd_to_extract.set_index([var1, var2]) 
            raster_interest.append(pd_to_extract)
    ready_to_plot=pd.concat(raster_interest)
    ready_to_plot.reset_index(inplace=True)
    return ready_to_plot


def generate_points_within_rectangles(x_values, y_values, width,height1, height2, num_points):
    """
    Generate extra points within rectangles centered at given points.

    Parameters:
    x_values (array-like): x-coordinates of the centroids.
    y_values (array-like): y-coordinates of the centroids.
    width (float): Width of the rectangles.
    height (float): Height of the rectangles.
    num_points (int): Number of points to generate within each rectangle.

    Returns:
    np.ndarray: Array of generated points with shape (num_points * len(x_values), 2).
    """
    points = []
    for x, y in zip(x_values, y_values):
        for _ in range(num_points):
            new_x = x + np.random.uniform(-height1, height2)
            new_y = y + np.random.uniform(-width, width)
            points.append([new_x, new_y])
    return np.array(points)

def calculate_speed(dif_x, dif_y, ts, number_frame_scene_changing=5):
    focal_distance_fbf = np.sqrt(np.sum([dif_x**2, dif_y**2], axis=0))
    focal_distance_fbf[0 : number_frame_scene_changing + 1] = (
        np.nan
    )  ##plus one to include the weird data from taking difference between 0 and some value
    instant_speed = focal_distance_fbf / np.diff(ts)
    return instant_speed


def time_series_plot(target_distance, instant_speed, angles, file_name, trial_id):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 7), tight_layout=True)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.size": 8})
    # Set the axis line width to 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["axes.linewidth"] = 2
    cmap = plt.get_cmap("viridis")
    ax1, ax2, ax3 = axes.flatten()
    ax1.set(title="Distance")
    ax2.set(title="Instant Speed")
    ax3.set(title="angular deviation")
    ax1.plot(np.arange(target_distance.shape[0]), target_distance)
    ax1.axhline(y=50,color="red",linestyle="--")
    ax2.plot(np.arange(instant_speed.shape[0]), instant_speed)
    ax2.axhline(y=1,color="red",linestyle="--")
    ax3.plot(np.arange(angles.shape[0]), angles)
    ax3.axhline(y=10,color="red",linestyle="--")
    fig_name = f"{file_name.stem.split('_')[0]}_{trial_id}_ts_plot.jpg"
    fig.savefig(file_name.parent / fig_name)

def behavioural_analysis(
    focal_xy, instant_speed, angular_velocity, follow_epochs, file_name, trial_id
):
    speed_threshold = 1
    walk_epochs = instant_speed > speed_threshold
    omega_threshold = 1
    turn_epochs = abs(angular_velocity) > omega_threshold
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4), tight_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.scatter(
        focal_xy[0, 1:],
        focal_xy[1, 1:],
        c="k",
        # c=np.zeros((1, focal_xy.shape[1] - 1), dtype=np.int8),
    )
    ax2.scatter(focal_xy[0, 1:][walk_epochs], focal_xy[1, 1:][walk_epochs], c="r")
    ax2.scatter(
        focal_xy[0, 1:][walk_epochs == False],
        focal_xy[1, 1:][walk_epochs == False],
        c="b",
        alpha=0.4,
    )
    ax3.scatter(focal_xy[0, 1:][turn_epochs], focal_xy[1, 1:][turn_epochs], c="m")
    ax3.scatter(
        focal_xy[0, 1:][turn_epochs == False],
        focal_xy[1, 1:][turn_epochs == False],
        c="b",
        alpha=0.4,
    )
    ax4.scatter(
        focal_xy[0, 1:][follow_epochs == True],
        focal_xy[1, 1:][follow_epochs == True],
        c="c",
    )
    ax4.scatter(
        focal_xy[0, 1:][follow_epochs == False],
        focal_xy[1, 1:][follow_epochs == False],
        c="b",
        alpha=0.4,
    )
    fig_name = (
        f"{file_name.stem.split('_')[0]}_{trial_id}_trajectory_analysis_speed{speed_threshold}.jpg"
    )
    fig.savefig(file_name.parent / fig_name)

def plot_trajectory(df_focal_animal, df_summary, df_agent, file_name):
    trajec_lim = 150
    variables = np.sort(
        df_summary[df_summary["type"] != "empty_trial"]["mu"].unique(), axis=0
    )
    fig, subplots = plt.subplots(
        nrows=1, ncols=variables.shape[0] + 1, figsize=(20, 4), tight_layout=True
    )
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.size": 8})
    # Set the axis line width to 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["axes.linewidth"] = 2
    # plt.rcParams['font.family'] = 'Helvetica'
    cmap = plt.get_cmap("viridis")
    for key, grp in df_summary.groupby("fname"):
        focal_xy = np.vstack(
            (
                df_focal_animal[df_focal_animal["fname"] == key]["X"].to_numpy(),
                df_focal_animal[df_focal_animal["fname"] == key]["Y"].to_numpy(),
            )
        )
        color = np.arange(focal_xy[0].shape[0])
        if grp["type"][0] == "empty_trial":
            subplot_title = "ISI"
            subplots[0].scatter(
                focal_xy[0],
                focal_xy[1],
                c=color,
                marker=".",
                alpha=0.5,
            )
            this_subplot = 0
        else:
            for count, this_variable in enumerate(variables):
                if this_variable == grp["mu"][0]:
                    this_subplot = count + 1
                    subplot_title = f"direction:{this_variable}"
                    subplots[this_subplot].scatter(
                        focal_xy[0],
                        focal_xy[1],
                        c=color,
                        marker=".",
                        alpha=0.5,
                    )
                    agent_xy=np.vstack((df_agent[df_agent["fname"] == key]["X"].to_numpy(),df_agent[df_agent["fname"] == key]["Y"].to_numpy()))
                    subplots[this_subplot].scatter(
                        agent_xy[0],
                        agent_xy[1],
                        c="k",
                        s=0.01,
                        marker=".",
                        alpha=0.1,
                    )
                else:
                    continue
        subplots[this_subplot].set(
            xlim=(-1 * trajec_lim, trajec_lim),
            ylim=(-1 * trajec_lim, trajec_lim),
            yticks=([-1 * trajec_lim, 0, trajec_lim]),
            xticks=([-1 * trajec_lim, 0, trajec_lim]),
            aspect=("equal"),
            title=subplot_title,
        )

    fig_name = f"{file_name.stem}_trajectory.jpg"
    fig.savefig(file_name.parent / fig_name)


def diff_angular_degree(angle_rad, number_frame_scene_changing,convert_unit=True):
    angle_rad[np.isnan(angle_rad)] = 0
    # angle_rad=np.unwrap(angle_rad)
    # ang_deg_diff=np.diff(np.unwrap(angle_rad))
    if convert_unit:
        ang_deg = np.mod(np.rad2deg(angle_rad), 360.0)
    else:
        #ang_deg = np.mod(angle_rad, 360.0)  ## if converting the unit to degree
        ang_deg=angle_rad
    ang_deg_diff = np.diff(
        np.unwrap(ang_deg, period=360)
    )  ##if converting the unit to degree
    angle_rad[0 : number_frame_scene_changing + 1] = (
        np.nan
    )  ##plus one to include the weird data from taking difference between 0 and some value
    ang_deg_diff[0 : number_frame_scene_changing + 1] = (
        np.nan
    )  ##plus one to include the weird data from taking difference between 0 and some value
    return angle_rad, ang_deg_diff


def classify_follow_epochs(
    focal_xy, instant_speed, ts, this_agent_xy, analysis_methods
):
    extract_follow_epoches = analysis_methods.get("extract_follow_epoches", True)
    follow_locustVR_criteria = analysis_methods.get("follow_locustVR_criteria", False)
    follow_within_distance = analysis_methods.get("follow_within_distance", 50)
    follow_above_speed = analysis_methods.get("follow_above_speed", 1)
    follow_within_angle= analysis_methods.get("follow_within_angle", 10)
    focal_distance_fbf = instant_speed * np.diff(ts)
    agent_distance_fbf = np.sqrt(
        np.sum([np.diff(this_agent_xy)[0] ** 2, np.diff(this_agent_xy)[1] ** 2], axis=0)
    )
    vector_dif = this_agent_xy - focal_xy
    target_distance = LA.norm(vector_dif, axis=0)
    dot_product = np.diag(
        np.matmul(np.transpose(np.diff(focal_xy)), np.diff(this_agent_xy))
    )
    angles = np.arccos(dot_product / focal_distance_fbf / agent_distance_fbf)
    angles_in_degree = angles * 180 / np.pi
    locustVR_criteria = (
        (target_distance[1:] < follow_within_distance)
        & (instant_speed > follow_above_speed)
        & (angles_in_degree < follow_within_angle)
    )
    walk_criteria = (target_distance[1:] < follow_within_distance) & (instant_speed > follow_above_speed)
    if extract_follow_epoches and follow_locustVR_criteria:
        epochs_of_interest = locustVR_criteria
    elif extract_follow_epoches:
        epochs_of_interest = walk_criteria
    else:
        epochs_of_interest = (
            np.ones((instant_speed.shape[0])) == 1.0
        )  # created a all-true array for overall heatmap
    return epochs_of_interest, vector_dif, angles_in_degree,walk_criteria


def align_agent_moving_direction(vector_dif, grp):
    theta = np.radians(grp["mu"].values[0] - 360)
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )  # calculate the rotation matrix to align the agent to move along the same direction
    vector_dif_rotated = rot_matrix @ vector_dif
    return vector_dif_rotated
    # vector_dif_rotated=vector_dif_rotated[:,1:]


def conclude_as_pd(
    df_focal_animal, vector_dif_rotated, epochs_of_interest, fname, agent_no):
    num_frames = df_focal_animal[df_focal_animal["fname"] == fname].shape[0]
    degree_in_the_trial = np.repeat(
        df_focal_animal[df_focal_animal["fname"] == fname]["mu"].to_numpy()[0],
        num_frames,
    )
    degree_time = np.vstack(
        (
            degree_in_the_trial,
            df_focal_animal[df_focal_animal["fname"] == fname]["ts"].to_numpy(),
        )
    )
    degree_time = degree_time[:, 1:]
    vector_dif_rotated = vector_dif_rotated[:, 1:]
    follow_wrap = np.concat(
        (vector_dif_rotated[:, epochs_of_interest], degree_time[:, epochs_of_interest])
    )
    follow_pd = pd.DataFrame(np.transpose(follow_wrap))
    follow_pd.insert(follow_pd.shape[1], "agent_id", np.repeat(agent_no, follow_pd.shape[0]))
    return follow_pd


def follow_behaviour_analysis(
    summary_file, focal_animal_file, agent_file, analysis_methods
):
    duration_for_baseline = 2
    pre_stim_ISI = 60
    trajec_lim = 150
    calculate_follow_chance_level = analysis_methods.get("calculate_follow_chance_level", False)
    agent_based_modeling=False
    variables_to_randomise='mu'
    randomise_heading_direction=False
    last_heading_direction_allocentric_view=False
    add_cumulated_angular_velocity=True
    analysis_window = analysis_methods.get("analysis_window")
    monitor_fps = analysis_methods.get("monitor_fps")
    align_with_isi_onset = analysis_methods.get("align_with_isi_onset", False)
    plotting_trajectory = analysis_methods.get("plotting_trajectory", False)
    plotting_event_distribution = analysis_methods.get("plotting_event_distribution", False)
    extract_follow_epoches = analysis_methods.get("extract_follow_epoches", True)
    follow_locustVR_criteria = analysis_methods.get("follow_locustVR_criteria", True)
    distribution_with_entire_body = analysis_methods.get("distribution_with_entire_body", True)

    df_agent_list = []
    with h5py.File(agent_file, "r") as f:
        for hdf_key in f.keys():
            tmp_agent = pd.read_hdf(agent_file, key=hdf_key)
            tmp_agent.insert(0, "type", np.repeat(hdf_key, tmp_agent.shape[0]))
            df_agent_list.append(tmp_agent)
    df_agent = pd.concat(df_agent_list)
    df_focal_animal = pd.read_hdf(focal_animal_file)
    df_summary = pd.read_hdf(summary_file)
    test = np.where(df_focal_animal["heading"].values == 0)[0]
    num_unfilled_gap = findLongestConseqSubseq(test, test.shape[0])
    print(f"the length :{num_unfilled_gap} of unfilled gap in {focal_animal_file}")
    if plotting_trajectory:
        plot_trajectory(df_focal_animal, df_summary, df_agent, focal_animal_file)
    dif_across_trials = []
    simulated_across_trials = []
    trial_evaluation_list = []
    raster_list = []
    trial_id = 0
    iteration_count = 0
    if analysis_methods.get("analyse_first_half_only", False):
        df_summary=df_summary[:int(df_summary.shape[0]/2)]
    elif analysis_methods.get("analyse_second_half_only", False):
        df_summary=df_summary[int(df_summary.shape[0]/2):]
    else:
        pass
    for key, grp in df_summary.groupby("fname"):
        focal_xy = np.vstack(
            (
                df_focal_animal[df_focal_animal["fname"] == key]["X"].to_numpy(),
                df_focal_animal[df_focal_animal["fname"] == key]["Y"].to_numpy(),
            )
        )
        dif_x = np.diff(focal_xy[0])
        dif_y = np.diff(focal_xy[1])
        ts = df_focal_animal[df_focal_animal["fname"] == key]["ts"].to_numpy()
        instant_speed = calculate_speed(dif_x, dif_y, ts)

        heading_direction = df_focal_animal[df_focal_animal["fname"] == key][
            "heading"
        ].to_numpy()
        _, turn_degree_fbf = diff_angular_degree(heading_direction, num_unfilled_gap)
        angular_velocity = turn_degree_fbf / np.diff(ts)
        if "density" in df_summary.columns:
        #if "type" in df_summary.columns:
            if align_with_isi_onset:
                if (
                    #df_focal_animal[df_focal_animal["fname"] == key]["density"][0]
                    grp['density'][0]==0.0
                ):
                    frame_range = analysis_window[1] * monitor_fps
                    xy_of_interest=focal_xy[:,:frame_range]
                    #d_of_interest = distance_from_centre[:frame_range]
                    v_of_interest = instant_speed[:frame_range]
                    w_of_interest = angular_velocity[:frame_range]
                    if "basedline_v" in locals():
                        normalised_v = v_of_interest / basedline_v
                    else:
                        normalised_v = np.repeat(np.nan, v_of_interest.shape[0])
                    if "basedline_w" in locals():
                        normalised_w = w_of_interest / basedline_w
                    else:
                        normalised_w = np.repeat(np.nan, w_of_interest.shape[0])
                else:
                    frame_range = analysis_window[0] * monitor_fps
                    xy_of_interest=focal_xy[:,frame_range:]
                    #d_of_interest = distance_from_centre[frame_range:]
                    v_of_interest = instant_speed[frame_range:]
                    w_of_interest = angular_velocity[frame_range:]
                    basedline_v = np.mean(
                        v_of_interest[-duration_for_baseline * monitor_fps :]
                    )
                    normalised_v = np.repeat(np.nan, v_of_interest.shape[0])
                    basedline_w = np.mean(
                        w_of_interest[-duration_for_baseline * monitor_fps :]
                    )
                    normalised_w = np.repeat(np.nan, w_of_interest.shape[0])
            else:
                if (
                    #df_focal_animal[df_focal_animal["fname"] == key]["density"][0]
                    grp['density'][0]==0.0
                ):
                    # print("ISI now")
                    frame_range = analysis_window[0] * monitor_fps
                    xy_of_interest=focal_xy[:,frame_range:]
                    #d_of_interest = distance_from_centre[frame_range:]
                    v_of_interest = instant_speed[frame_range:]
                    w_of_interest = angular_velocity[frame_range:]
                    basedline_v = np.mean(
                        v_of_interest[-duration_for_baseline * monitor_fps :]
                    )
                    normalised_v = np.repeat(np.nan, v_of_interest.shape[0])
                    last_w=w_of_interest[-duration_for_baseline * monitor_fps :]
                    basedline_w = np.mean(last_w)#### need to double check here
                    normalised_w = np.repeat(np.nan, w_of_interest.shape[0])
                    last_heading = circmean(heading_direction[-duration_for_baseline * monitor_fps :])

                else:
                    if "basedline_v" in locals():
                        normalised_v = v_of_interest / basedline_v
                    else:
                        normalised_v = np.repeat(np.nan, v_of_interest.shape[0])
                    if "basedline_w" in locals():
                        normalised_w = w_of_interest / basedline_w
                    else:
                        normalised_w = np.repeat(np.nan, w_of_interest.shape[0])
        else:
            if align_with_isi_onset:
                if grp["type"][0] == "empty_trial":
                    frame_range = analysis_window[1] * monitor_fps
                    xy_of_interest=focal_xy[:,:frame_range]
                    #d_of_interest = distance_from_centre[:frame_range]
                    v_of_interest = instant_speed[:frame_range]
                    w_of_interest = angular_velocity[:frame_range]
                else:
                    frame_range = analysis_window[0] * monitor_fps
                    xy_of_interest=focal_xy[:,frame_range:]
                    #d_of_interest = distance_from_centre[frame_range:]
                    v_of_interest = instant_speed[frame_range:]
                    w_of_interest = angular_velocity[frame_range:]
            else:
                if grp["type"][0] == "empty_trial":
                    # print("ISI now")if "density" in grp.columns and grp["density"][0] == 0:
                    frame_range = analysis_window[0] * monitor_fps
                    xy_of_interest=focal_xy[:,frame_range:]
                    #d_of_interest = distance_from_centre[frame_range:]
                    v_of_interest = instant_speed[frame_range:]
                    w_of_interest = angular_velocity[frame_range:]
                    basedline_v = np.mean(
                        v_of_interest[-duration_for_baseline * monitor_fps :]
                    )
                    normalised_v = np.repeat(np.nan, v_of_interest.shape[0])
                    last_w=w_of_interest[-duration_for_baseline * monitor_fps :]
                    basedline_w = np.mean(last_w)#### need to double check here
                    normalised_w = np.repeat(np.nan, w_of_interest.shape[0])
                    last_heading = circmean(heading_direction[-duration_for_baseline * monitor_fps :])
                else:
                    # print("stim now")
                    frame_range = analysis_window[1] * monitor_fps
                    #d_of_interest = distance_from_centre[:frame_range]
                    xy_of_interest=focal_xy[:,:frame_range]
                    v_of_interest = instant_speed[:frame_range]
                    w_of_interest = angular_velocity[:frame_range]
                    if "basedline_v" in locals():
                        normalised_v = v_of_interest / basedline_v
                    else:
                        normalised_v = np.repeat(np.nan, v_of_interest.shape[0])
                    if "basedline_w" in locals():
                        normalised_w = w_of_interest / basedline_w
                    else:
                        normalised_w = np.repeat(np.nan, w_of_interest.shape[0])


        if "density" in df_summary.columns:
            con_matrex = (
                #d_of_interest,
                xy_of_interest[0,:],
                xy_of_interest[1,:],
                v_of_interest,
                w_of_interest,
                normalised_v,
                normalised_w,
                np.repeat(iteration_count, v_of_interest.shape[0]),
                np.repeat(grp["mu"][0], v_of_interest.shape[0]),
                np.repeat(grp["density"][0], v_of_interest.shape[0]),
            )
        else:
            con_matrex = (
                #d_of_interest,
                xy_of_interest[0,:],
                xy_of_interest[1,:],
                v_of_interest,
                w_of_interest,
                normalised_v,
                normalised_w,
                np.repeat(iteration_count, v_of_interest.shape[0]),
                np.repeat(grp["mu"][0], v_of_interest.shape[0]),
                np.repeat(grp["type"][0], v_of_interest.shape[0]),
            )
        raster_list.append(pd.DataFrame(np.transpose(np.vstack(con_matrex))))
        iteration_count += 1
        if "density" in grp.columns and grp["density"][0] == 0:
            focal_distance_ISI = instant_speed * np.diff(ts)
            _, turn_degree_ISI = diff_angular_degree(
                heading_direction, num_unfilled_gap
            )
            pre_stim_ISI = grp["duration"][0]
            continue
        elif grp["type"][0] == "empty_trial":
            focal_distance_ISI = instant_speed * np.diff(ts)
            _, turn_degree_ISI = diff_angular_degree(
                heading_direction, num_unfilled_gap
            )
            pre_stim_ISI = grp["duration"][0]
            continue
        else:
            focal_distance_fbf = instant_speed * np.diff(ts)
            agent_xy = np.vstack(
                (
                    df_agent[df_agent["fname"] == key]["X"].to_numpy(),
                    df_agent[df_agent["fname"] == key]["Y"].to_numpy(),
                )
            )
            follow_pd_list = []
            simulated_pd_list=[]
            if calculate_follow_chance_level and agent_based_modeling:
                if randomise_heading_direction:
                    simulated_heading=np.random.uniform(-0.5,0.5,1)*np.pi
                elif last_heading_direction_allocentric_view:
                    simulated_heading=last_heading
                else:
                    simulated_heading=0
                if add_cumulated_angular_velocity:
                    simulated_angular_velocity=np.concatenate((last_w,np.zeros(ts.shape[0]-duration_for_baseline*monitor_fps)))#or can replace last_w with basedline_w*np.ones(duration_for_baseline*monitor_fps))
                    heading_change_fbf=simulated_angular_velocity[:np.diff(ts).shape[0]]*np.diff(ts)
                    simulated_heading_arr=simulated_heading*np.ones(ts.shape[0])+np.cumsum(np.append(heading_change_fbf, heading_change_fbf[heading_change_fbf.shape[0]-1]))
                else:
                    simulated_heading_arr=simulated_heading*np.ones(ts.shape[0])
                #can apply previous angular velocity in the first 2 second here to make it more realistic
                simulated_x=np.cumsum(basedline_v*np.cos(simulated_heading_arr))/monitor_fps
                simulated_y=np.cumsum(basedline_v*np.sin(simulated_heading_arr))/monitor_fps
                simulated_speed=calculate_speed(np.diff(simulated_x), np.diff(simulated_y), ts)
                
            if agent_xy.shape[1] > focal_xy.shape[1]:
                num_agent = round(agent_xy.shape[1] / focal_xy.shape[1])
            else:
                num_agent = 1
                # Loop through the array in two portions
            agent_split=np.array_split(agent_xy,num_agent,axis=1)
            for i in range(num_agent):
                this_agent_xy=agent_split[i]
                if np.isnan(np.min(agent_xy)) == True:
                    ##remove nan from agent's xy with interpolation
                    tmp_arr_x = this_agent_xy[0]
                    tmp_arr_y = this_agent_xy[1]
                    nans, x = nan_helper(tmp_arr_x)
                    tmp_arr_x[nans] = np.interp(x(nans), x(~nans), tmp_arr_x[~nans])
                    nans, y = nan_helper(tmp_arr_y)
                    tmp_arr_y[nans] = np.interp(y(nans), y(~nans), tmp_arr_y[~nans])
                    this_agent_xy=np.vstack((tmp_arr_x,tmp_arr_y))
                epochs_of_interest, vector_dif, angles_in_degree,walk_epochs = (
                classify_follow_epochs(focal_xy, instant_speed, ts, this_agent_xy, analysis_methods))
                vector_dif_rotated = align_agent_moving_direction(vector_dif, grp)
                ### Here is a quick fix to analyse agents in a marching band.
                ### instead of assigning agents ID based on which agent is analysed first (like in the 2-choice assay)
                ### this one assign agents ID or more precisely band ID, based on whether the agents is in the left or right side from the allocentric view
                ### this means if the agent's coordinate is negative, we assume this agent is on the right. Otherwise, on the left side, and given then ID accordingly.
                ### However, if experiment are more advanced, for example, a locust navigate in a marching band mixed with different type of agent. This analysis will fail
                ### Then we basically needs to change the code in locustvr_converter.py and assign ID for each type of agents explicitly.
                if num_agent>2 and this_agent_xy[1,:].mean()>0:
                    follow_pd = conclude_as_pd(df_focal_animal, vector_dif_rotated, epochs_of_interest, key, 1)
                elif num_agent>2 and this_agent_xy[1,:].mean()<0:
                    follow_pd = conclude_as_pd(df_focal_animal, vector_dif_rotated, epochs_of_interest, key, 0)
                else:
                    follow_pd = conclude_as_pd(df_focal_animal, vector_dif_rotated, epochs_of_interest, key, i)
                follow_pd.insert(follow_pd.shape[1],"type",
                        np.repeat(
                            df_agent[df_agent["fname"] == key]["type"].values[0],
                            follow_pd.shape[0],
                        ))
                if "rotation_gain" in df_summary.columns:
                    follow_pd.insert(follow_pd.shape[1],"rotation_gain",
                            np.repeat(
                                grp["rotation_gain"].values[0],
                                follow_pd.shape[0],
                            ))
                    follow_pd.insert(follow_pd.shape[1],"translation_gain",
                            np.repeat(
                                grp["translation_gain"].values[0],
                                follow_pd.shape[0],
                            ))
                    
                if calculate_follow_chance_level and agent_based_modeling:
                    epochs_by_chance,simulated_vector,_,_=classify_follow_epochs(
                            np.vstack((simulated_x,simulated_y)), simulated_speed, ts, this_agent_xy, analysis_methods
                    )
                    vector_dif_simulated = align_agent_moving_direction(simulated_vector, grp)
                    simulated_pd= conclude_as_pd(
                            df_focal_animal, vector_dif_simulated, epochs_by_chance, key, i
                    )
                elif calculate_follow_chance_level and df_summary[variables_to_randomise].unique().shape[0]>1:
                    #other_vars=df_summary[variables_to_randomise].unique()[grp[variables_to_randomise][0]!=df_summary[variables_to_randomise].unique()]
                    #permutation_vars=other_vars
                    permutation_vars=df_summary[variables_to_randomise].unique()### this is used to calculate shuffled variables. Depends on the trial exchangability, it can be called as permutation test sometimes
                    probabilities=np.ones(permutation_vars.shape)/permutation_vars.shape[0]
                    #probabilities = [1/3, 1/3, 1/3]
                    shuffled_mu=np.random.choice(permutation_vars, 1, p=probabilities)
                    # b=np.random.binomial(n=1,p=0.5,size=1)
                    # if b==0:
                    #     theta = np.radians(-45)
                    # else:
                    #     theta = np.radians(45)
                    theta = np.radians(shuffled_mu[0])
                    rot_matrix = np.array(
                            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
                        )  # calculate the rotation matrix to align the agent to move along the same direction
                    this_agent_xy_rotated = rot_matrix @ this_agent_xy
                    epochs_by_chance,simulated_vector,_,_=classify_follow_epochs(focal_xy, instant_speed, ts, this_agent_xy_rotated, analysis_methods)
                    vector_dif_simulated = align_agent_moving_direction(simulated_vector, grp)
                    ## same situation as line 617
                    if num_agent>2 and this_agent_xy_rotated[1,:].mean()>0:
                        simulated_pd = conclude_as_pd(df_focal_animal, vector_dif_simulated, epochs_by_chance, key, 1)
                    elif num_agent>2 and this_agent_xy_rotated[1,:].mean()<0:
                        simulated_pd = conclude_as_pd(df_focal_animal, vector_dif_simulated, epochs_by_chance, key, 0)
                    else:
                        simulated_pd= conclude_as_pd(
                                df_focal_animal, vector_dif_simulated, epochs_by_chance, key, i
                        )

                    simulated_pd.insert(
                            simulated_pd.shape[1],
                            "type",
                            np.repeat(
                                df_agent[df_agent["fname"] == key]["type"].values[0],
                                simulated_pd.shape[0],
                            ),
                    )
                    ## should add translational and rotational gain in simulated_pd too but lets figure out a better way to do this first
                if plotting_trajectory:
                    target_distance = LA.norm(vector_dif, axis=0)
                    time_series_plot(
                            target_distance,
                            instant_speed,
                            angles_in_degree,
                            focal_animal_file,
                            key,
                    )
                follow_pd_list.append(follow_pd)
                if "simulated_pd" in locals():
                    simulated_pd_list.append(simulated_pd)
            if num_agent>1:
                follow_pd_combined = pd.concat(follow_pd_list)
                dif_across_trials.append(follow_pd_combined)
                if 'ts' in follow_pd_combined:
                    num_duplocated_epochs = follow_pd_combined.duplicated(subset=['ts']).sum()
                else:
                    num_duplocated_epochs = follow_pd_combined.duplicated(3).sum()
                sum_follow_epochs = follow_pd_combined.shape[0]-num_duplocated_epochs
                
                if 'simulated_pd_list' in locals() and len(simulated_pd_list)>0:
                    simulated_pd_combined = pd.concat(simulated_pd_list)
                    simulated_across_trials.append(simulated_pd_combined)
                    num_duplocated_epochs = simulated_pd_combined.duplicated(3).sum()
                    sum_chance_epochs = simulated_pd_combined.shape[0]-num_duplocated_epochs
                else:
                    sum_chance_epochs=np.nan
            else:
                dif_across_trials.append(follow_pd)
                sum_follow_epochs = follow_pd.shape[0]
                if 'simulated_pd' in locals():
                    simulated_across_trials.append(simulated_pd)
                    sum_chance_epochs = simulated_pd.shape[0]
                else:
                    sum_chance_epochs=np.nan
            # _, turn_degree_fbf = diff_angular_degree(
            #     heading_direction, num_unfilled_gap
            # )
            # angular_velocity = turn_degree_fbf / np.diff(ts)
            trial_summary = pd.DataFrame(
                {
                    "trial_id": [trial_id],
                    "mu": [grp["mu"].values[0]],
                    "polar_angle": [grp["polar_angle"].values[0]],
                    "radial_distance": [grp["radial_distance"].values[0]],
                    "speed": [grp["speed"][0]],
                    "num_follow_epochs": [sum_follow_epochs],
                    "num_chance_epochs": [sum_chance_epochs],
                    "num_walk_epochs":[sum(walk_epochs)],
                    "number_frames": [focal_xy.shape[1] - 1],
                    "travel_distance": [np.nansum(focal_distance_fbf)],
                    "total_turning": [np.nansum(abs(turn_degree_fbf))],
                    "gross_turning": [np.nansum(turn_degree_fbf)],
                    "travel_distance_ISI": [np.nansum(focal_distance_ISI)],
                    "total_turning_ISI": [np.nansum(abs(turn_degree_ISI))],
                    "gross_turning_ISI": [np.nansum(turn_degree_ISI)],
                    "duration": [grp["duration"].values[0]],
                    "duration_ISI": [pre_stim_ISI],
                    "temperature": [
                        df_focal_animal[df_focal_animal["fname"] == key][
                            "temperature"
                        ].values[0]
                    ],
                    "humidity": [
                        df_focal_animal[df_focal_animal["fname"] == key][
                            "humidity"
                        ].values[0]
                    ],
                    "object": [grp["type"].values[0]],
                }
            )
            if "rotation_gain" in df_summary.columns:
                trial_summary['rotation_gain']=[grp["rotation_gain"].values[0]]
                trial_summary['translation_gain']=[grp["translation_gain"].values[0]]
            if "density" in df_summary.columns:
                trial_summary['density'] = grp["density"][0]
            trial_evaluation_list.append(trial_summary)
            trial_id = trial_id + 1
    raster_pd = pd.concat(raster_list)
    if "density" in df_summary.columns:
        raster_pd.columns = [
            "X",
            "Y",
            "velocity",
            "omega",
            "normalised_v",
            "normalised_omega",
            "id",
            "mu",
            "density",
        ]
    elif "type" in df_summary.columns:
        raster_pd.columns = [
            "X",
            "Y",
            "velocity",
            "omega",
            "normalised_v",
            "normalised_omega",
            "id",
            "mu",
            "object",
        ]
    else:
        raster_pd.columns = [
            "X",
            "Y",
            "velocity",
            "omega",
            "normalised_v",
            "normalised_omega",
            "id",
            "mu",
        ]

    dif_across_trials_pd = pd.concat(dif_across_trials)
    if 'simulated_across_trials' in locals() and len(simulated_across_trials)>0:
        simulated_across_trials_pd = pd.concat(simulated_across_trials)
    if "rotation_gain" in df_summary.columns:
        dif_column_list = ["x", "y", "degree", "ts","agent_id","type","rotation_gain","translation_gain"]
        dif_across_trials_pd.columns=dif_column_list[:dif_across_trials_pd.shape[1]]
    else:
        dif_column_list = ["x", "y", "degree", "ts","agent_id","type"]
        dif_across_trials_pd.columns=dif_column_list[:dif_across_trials_pd.shape[1]]
    if 'simulated_across_trials_pd' in locals():
        simulated_across_trials_pd.columns=dif_column_list[:simulated_across_trials_pd.shape[1]]
    else:
        simulated_across_trials_pd=[]
    trial_evaluation_pd=pd.concat(trial_evaluation_list)
    this_animal_follow_ratio=trial_evaluation_pd['num_follow_epochs'].sum()/trial_evaluation_pd['number_frames'].sum()
    print(f"the follow ratio of {summary_file.stem.split('_')[0]} in {summary_file.parent} is {this_animal_follow_ratio}")

    if plotting_event_distribution:
        plt.close()
        trial_evaluation=pd.concat(trial_evaluation_list)
        trial_evaluation['time_group'] = pd.cut(trial_evaluation['trial_id'], bins=3, labels=[1, 2, 3])
        fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(9,5), tight_layout=True)
        ax, ax2 = axes.flatten()
        test1=ax.scatter(x=trial_evaluation['trial_id'],y=trial_evaluation['num_follow_epochs'],c=trial_evaluation['travel_distance'],cmap='viridis',s=10)
        ax.set_xticks([])
        ax.set_xlabel("Trial#")
        ax.set(
            ylabel="Number of follow epochs",
            xticks=[0, trial_evaluation['trial_id'].max()],
            xlabel="Trial#",
        )
        plt.colorbar(test1,label="Travel distance (cm)")
        test2=ax2.scatter(x=trial_evaluation['time_group'],y=trial_evaluation['travel_distance'],c=trial_evaluation['mu'],cmap='Set1',s=10)
        ax2.set(
            ylabel="Travel distance (cm)",
            xticks=[1,2,3],
            xlabel="n-th part of the trials",
        )
        legend2 = ax2.legend(*test2.legend_elements(), title="Degree")
        ax2.add_artist(legend2)
        fig_name = (
                f"{summary_file.stem.split('_')[0]}_follow_across_time.jpg"
            )
        fig.savefig(summary_file.parent / fig_name)
        if extract_follow_epoches and follow_locustVR_criteria:
            xlimit=(0,15)
            ylimit=(-5,5)
        elif extract_follow_epoches:
            xlimit=(0,40)
            ylimit=(-15,15)
        else:
            xlimit=(-20,100)
            ylimit=(-45,45)
        for keys, grp in dif_across_trials_pd.groupby(['type','degree']):
            fig = plt.figure(figsize=(9,5))
            ax = fig.add_subplot(212)
            ax2 = fig.add_subplot(221)
            ax3 = fig.add_subplot(222)
            ax.hist(grp['ts'].values,bins=100,density=False,color='r')
            if len(simulated_across_trials_pd)>0:
                sim_grp=simulated_across_trials_pd[(simulated_across_trials_pd['type']==keys[0])&(simulated_across_trials_pd['degree']==keys[1])]
                ax.hist(sim_grp['ts'].values,bins=100,density=False,color='tab:gray',alpha=0.3)
            else:
                sim_grp=np.array([])
            #ax.set(xlim=(0,60),ylim=(0,0.05),title=f'agent:{keys[0]},deg:{int(keys[1])}')
            ax.set(xlim=(0,60),title=f'agent:{keys[0]},deg:{int(keys[1])}')
            if distribution_with_entire_body:
                if grp.shape[0]>0:
                    body_points=generate_points_within_rectangles(grp['x'].values,grp['y'].values, 1,4,2,21)
                    ax2.hist2d(body_points[:,0],body_points[:,1],bins=400)
                if sim_grp.shape[0]>0:
                    body_points=generate_points_within_rectangles(sim_grp['x'].values,sim_grp['y'].values, 1,4,2,21)
                    ax3.hist2d(body_points[:,0],body_points[:,1],bins=400)
            else:
                if grp.shape[0]>0:
                    ax2.hist2d(grp['x'].values,grp['y'].values,bins=100)
                if sim_grp.shape[0]>0:
                    ax3.hist2d(sim_grp['x'].values,sim_grp['y'].values,bins=100)
            ax2.set(
            yticks=[ylimit[0],ylimit[1]],
            xticks=[xlimit[0],xlimit[1]],
            xlim=xlimit,ylim=ylimit,adjustable='box', aspect='equal')
            ax3.set(
            yticks=[ylimit[0],ylimit[1]],
            xticks=[xlimit[0],xlimit[1]],
            xlim=xlimit,ylim=ylimit,adjustable='box', aspect='equal')
            '''
            ax.axvline(
                    x=6,
                    color="red",
                    linestyle="--",
                    label="length")
            ax.axvline(
                    x=0,
                    color="red",
                    linestyle="--",
                    label="length")
            ax.axhline(
                    y=-1,
                    color="red",
                    linestyle="--",
                    label="width")
            ax.axhline(
                    y=1,
                    color="red",
                    linestyle="--",
                    label="width")
            '''
            #ax2.hist(grp['ts'].values,bins=100,density=True,histtype="step",cumulative=True,label="Cumulative histogram")

            fig_name = (
                f"{summary_file.stem.split('_')[0]}_{keys[0]}_{keys[1]}_follow_distribution.jpg"
            )
            fig.savefig(summary_file.parent / fig_name)
    return dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap, simulated_across_trials_pd


def load_data(this_dir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            #print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())

    agent_pattern = f"VR3*agent_full.h5"
    agent_file = find_file(Path(this_dir), agent_pattern)
    xy_pattern = f"VR3*XY_full.h5"
    focal_animal_file = find_file(Path(this_dir), xy_pattern)
    summary_pattern = f"VR3*score_full.h5"
    summary_file = find_file(Path(this_dir), summary_pattern)

    dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap,simulated_across_trials_pd = (
        follow_behaviour_analysis(
            summary_file, focal_animal_file, agent_file, analysis_methods
        )
    )
    return dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap,simulated_across_trials_pd


if __name__ == "__main__":
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241125_131510"
    thisDir = r"D:\MatrexVR_2024_3_Data\RunData\20250709_155715"
    #thisDir = r"D:\MatrexVR_2024_Data\RunData\20250523_143428"
    #thisDir = r"D:/MatrexVR_grass1_Data/RunData/20240907_190839"
    #thisDir = r"D:/MatrexVR_grass1_Data/RunData/20240907_142802"
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241110_165438"
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241116_134457"
    #thisDir = r"C:\Users\neuroLaptop\Documents\MatrexVR_2024_Data\RunData\20241116_134457"
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241225_134852"
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20250423_112912"
    #thisDir =r"D:/MatrexVR_2024_Data/RunData/20241231_130927"
    # thisDir = r"D:/MatrexVR_2024_Data/RunData/20241201_131605"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
