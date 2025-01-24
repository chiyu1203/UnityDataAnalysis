import time, sys, json
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
from data_cleaning import findLongestConseqSubseq
colormap_name = "viridis"
sm = cm.ScalarMappable(cmap=colormap_name)

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
    ax2.axhline(y=50,color="red",linestyle="--")
    ax2.plot(np.arange(instant_speed.shape[0]), instant_speed)
    ax2.axhline(y=1,color="red",linestyle="--")
    ax3.plot(np.arange(angles.shape[0]), angles)
    ax3.axhline(y=10,color="red",linestyle="--")
    fig_name = f"{file_name.stem.split('_')[0]}_{trial_id}_ts_plot.jpg"
    fig.savefig(file_name.parent / fig_name)
    fig.show()


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
        f"{file_name.stem.split('_')[0]}_{trial_id}_trajectory_analysis_speed1.jpg"
    )
    fig.savefig(file_name.parent / fig_name)
    fig.show()


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
                    subplots[this_subplot].plot(
                        df_agent[df_agent["fname"] == key]["X"].to_numpy(),
                        df_agent[df_agent["fname"] == key]["Y"].to_numpy(),
                        c="k",
                        # marker=".",
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


def diff_angular_degree(angle_rad, number_frame_scene_changing):
    angle_rad[np.isnan(angle_rad)] = 0
    # angle_rad=np.unwrap(angle_rad)
    # ang_deg_diff=np.diff(np.unwrap(angle_rad))
    ang_deg = np.mod(np.rad2deg(angle_rad), 360.0)  ## if converting the unit to degree
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
        & (instant_speed > 1)
        & (angles_in_degree < 10)
    )
    walk_criteria = (target_distance[1:] < follow_within_distance) & (instant_speed > 1)
    if extract_follow_epoches and follow_locustVR_criteria:
        epochs_of_interest = locustVR_criteria
    elif extract_follow_epoches:
        epochs_of_interest = walk_criteria
    else:
        epochs_of_interest = (
            np.ones((instant_speed.shape[0])) == 1.0
        )  # created a all-true array for overall heatmap
    return epochs_of_interest, vector_dif, angles_in_degree


def align_agent_moving_direction(vector_dif, grp):
    theta = np.radians(grp["mu"].values[0] - 360)
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )  # calculate the rotation matrix to align the agent to move along the same direction
    vector_dif_rotated = rot_matrix @ vector_dif
    return vector_dif_rotated
    # vector_dif_rotated=vector_dif_rotated[:,1:]


def conclude_as_pd(
    df_focal_animal, vector_dif_rotated, epochs_of_interest, fname, agent_no=0
):
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
    follow_pd.insert(0, "agent_id", np.repeat(agent_no, follow_pd.shape[0]))
    return follow_pd


def calculate_relative_position(
    summary_file, focal_animal_file, agent_file, analysis_methods
):
    duration_for_baseline = 2
    pre_stim_ISI = 60
    trajec_lim = 150
    randomise_heading_direction=False
    last_heading_direction_allocentric_view=False
    analysis_window = analysis_methods.get("analysis_window")
    monitor_fps = analysis_methods.get("monitor_fps")
    align_with_isi_onset = analysis_methods.get("align_with_isi_onset", False)
    plotting_trajectory = analysis_methods.get("plotting_trajectory", False)
    plotting_event_distribution = analysis_methods.get("plotting_event_distribution", False)
    extract_follow_epoches = analysis_methods.get("extract_follow_epoches", True)
    follow_locustVR_criteria = analysis_methods.get("follow_locustVR_criteria", True)
    distribution_with_entire_body = analysis_methods.get("distribution_with_entire_body", True)
    calculate_follow_chance_level = analysis_methods.get("calculate_follow_chance_level", True)
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
        distance_from_centre = np.sqrt(
            np.sum([focal_xy[0] ** 2, focal_xy[1] ** 2], axis=0)
        )
        # angle_rad = df_focal_animal[df_focal_animal["fname"]==key]["heading"].to_numpy()
        _, turn_degree_fbf = diff_angular_degree(heading_direction, num_unfilled_gap)
        angular_velocity = turn_degree_fbf / np.diff(ts)
        if "type" in df_summary.columns:
            if align_with_isi_onset:
                if grp["type"][0] == "empty_trial":
                    frame_range = analysis_window[1] * monitor_fps
                    d_of_interest = distance_from_centre[:frame_range]
                    v_of_interest = instant_speed[:frame_range]
                    w_of_interest = angular_velocity[:frame_range]
                else:
                    frame_range = analysis_window[0] * monitor_fps
                    d_of_interest = distance_from_centre[frame_range:]
                    v_of_interest = instant_speed[frame_range:]
                    w_of_interest = angular_velocity[frame_range:]
            else:
                if grp["type"][0] == "empty_trial":
                    # print("ISI now")
                    frame_range = analysis_window[0] * monitor_fps
                    d_of_interest = distance_from_centre[frame_range:]
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
                    last_heading = circmean(heading_direction[-duration_for_baseline * monitor_fps :])
                else:
                    # print("stim now")
                    frame_range = analysis_window[1] * monitor_fps
                    d_of_interest = distance_from_centre[:frame_range]
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
            if align_with_isi_onset:
                if (
                    df_focal_animal[df_focal_animal["fname"] == key]["density"][0]
                    == 0.0
                ):
                    frame_range = analysis_window[1] * monitor_fps
                    d_of_interest = distance_from_centre[:frame_range]
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
                    d_of_interest = distance_from_centre[frame_range:]
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
                    df_focal_animal[df_focal_animal["fname"] == key]["density"][0]
                    == 0.0
                ):
                    # print("ISI now")
                    frame_range = analysis_window[0] * monitor_fps
                    d_of_interest = distance_from_centre[frame_range:]
                    v_of_interest = instant_speed[frame_range:]
                    w_of_interest = angular_velocity[frame_range:]

                else:
                    # print("Stim now")
                    frame_range = analysis_window[1] * monitor_fps
                    d_of_interest = distance_from_centre[:frame_range]
                    v_of_interest = instant_speed[:frame_range]
                    w_of_interest = angular_velocity[:frame_range]

        if "type" in df_summary.columns:
            con_matrex = (
                d_of_interest,
                v_of_interest,
                w_of_interest,
                normalised_v,
                normalised_w,
                np.repeat(iteration_count, v_of_interest.shape[0]),
                np.repeat(grp["mu"][0], v_of_interest.shape[0]),
                np.repeat(grp["type"][0], v_of_interest.shape[0]),
            )
        else:
            con_matrex = (
                d_of_interest,
                v_of_interest,
                w_of_interest,
                normalised_v,
                normalised_w,
                np.repeat(iteration_count, v_of_interest.shape[0]),
                np.repeat(
                    df_focal_animal[df_focal_animal["fname"] == key]["mu"][0],
                    v_of_interest.shape[0],
                ),
                np.repeat(
                    df_focal_animal[df_focal_animal["fname"] == key]["density"][0],
                    v_of_interest.shape[0],
                ),
            )
        # raw_data=np.vstack(con_matrex)
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
            if calculate_follow_chance_level:
                if randomise_heading_direction:
                    simulated_heading=np.random.uniform(-0.5,0.5,1)*np.pi
                elif last_heading_direction_allocentric_view:
                    simulated_heading=last_heading
                else:
                    simulated_heading=0
                simulated_x=np.cumsum(basedline_v*np.cos(simulated_heading)*np.ones(ts.shape[0]))/monitor_fps
                simulated_y=np.cumsum(basedline_v*np.sin(simulated_heading)*np.ones(ts.shape[0]))/monitor_fps
                simulated_speed=calculate_speed(np.diff(simulated_x), np.diff(simulated_y), ts)
            if np.isnan(np.min(agent_xy)) == True:
                ##remove nan from agent's xy with interpolation
                tmp_arr = agent_xy[0]
                tmp_arr1 = agent_xy[1]
                nans, x = nan_helper(tmp_arr)
                tmp_arr[nans] = np.interp(x(nans), x(~nans), tmp_arr[~nans])
                nans, y = nan_helper(tmp_arr1)
                tmp_arr1[nans] = np.interp(y(nans), y(~nans), tmp_arr1[~nans])
            if agent_xy.shape[1] > focal_xy.shape[1]:
                num_portion = round(agent_xy.shape[1] / focal_xy.shape[1])
                midpoint = agent_xy.shape[1] // num_portion
                # Loop through the array in two portions
                follow_pd_list = []
                simulated_pd_list=[]
                for i in range(num_portion):
                    if i == 0:
                        this_agent_xy = agent_xy[:, :midpoint]  # First half
                        # print(f"Processing first half: {this_agent_xy}")
                    else:
                        this_agent_xy = agent_xy[:, midpoint:]  # Second half
                        # print(f"Processing second half: {this_agent_xy}")
                    epochs_of_interest, vector_dif, angles_in_degree = (
                        classify_follow_epochs(
                            focal_xy, instant_speed, ts, this_agent_xy, analysis_methods
                        )
                    )
                    vector_dif_rotated = align_agent_moving_direction(vector_dif, grp)
                    follow_pd = conclude_as_pd(
                        df_focal_animal, vector_dif_rotated, epochs_of_interest, key, i
                    )
                    follow_pd.insert(
                        0,
                        "type",
                        np.repeat(
                            df_agent[df_agent["fname"] == key]["type"].values[0],
                            follow_pd.shape[0],
                        ),
                    )
                    if calculate_follow_chance_level:
                        epochs_by_chance,simulated_vector,_=classify_follow_epochs(
                            np.vstack((simulated_x,simulated_y)), simulated_speed, ts, this_agent_xy, analysis_methods
                        )
                        vector_dif_simulated = align_agent_moving_direction(simulated_vector, grp)
                        simulated_pd= conclude_as_pd(
                            df_focal_animal, vector_dif_simulated, epochs_by_chance, key, i
                        )
                        simulated_pd.insert(
                            0,
                            "type",
                            np.repeat(
                                df_agent[df_agent["fname"] == key]["type"].values[0],
                                simulated_pd.shape[0],
                            ),
                        )
                        simulated_pd_list.append(simulated_pd)
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
            else:
                epochs_of_interest, vector_dif, angles_in_degree = (
                    classify_follow_epochs(
                        focal_xy, instant_speed, ts, agent_xy, analysis_methods
                    )
                )
                vector_dif_rotated = align_agent_moving_direction(vector_dif, grp)
                follow_pd = conclude_as_pd(
                    df_focal_animal, vector_dif_rotated, epochs_of_interest, key
                )
                follow_pd.insert(
                    0,
                    "type",
                    np.repeat(
                        df_agent[df_agent["fname"] == key]["type"].values[0],
                        follow_pd.shape[0],
                    ),
                )
                if calculate_follow_chance_level:
                    epochs_by_chance,simulated_vector,_=classify_follow_epochs(
                        np.vstack((simulated_x,simulated_y)), simulated_speed, ts, agent_xy, analysis_methods
                    )
                    vector_dif_simulated = align_agent_moving_direction(simulated_vector, grp)
                    simulated_pd= conclude_as_pd(
                        df_focal_animal, vector_dif_simulated, epochs_by_chance, key
                    )
                    simulated_pd.insert(
                        0,
                        "type",
                        np.repeat(
                            df_agent[df_agent["fname"] == key]["type"].values[0],
                            simulated_pd.shape[0],
                        ),
                    )
                if plotting_trajectory:
                    target_distance = LA.norm(vector_dif, axis=0)
                    time_series_plot(
                        target_distance,
                        instant_speed,
                        angles_in_degree,
                        focal_animal_file,
                        key,
                    )
            if "follow_pd_list" in locals():
                follow_pd_combined = pd.concat(follow_pd_list)
                dif_across_trials.append(follow_pd_combined)
                sum_follow_epochs = follow_pd_combined.shape[0]
                if 'simulated_pd_list' in locals():
                    simulated_pd_combined = pd.concat(simulated_pd_list)
                    simulated_across_trials.append(simulated_pd_combined)
                    sum_chance_epochs = simulated_pd_combined.shape[0]
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
                    # "this_vr": [grp['this_vr'][0]],
                    "num_follow_epochs": [sum_follow_epochs],
                    "num_chance_epochs": [sum_chance_epochs],
                    "number_frames": [focal_xy.shape[1] - 1],
                    "travel_distance": [np.nansum(focal_distance_fbf)],
                    "turning_distance": [np.nansum(abs(turn_degree_fbf))],
                    "travel_distance_ISI": [np.nansum(focal_distance_ISI)],
                    "turning_distance_ISI": [np.nansum(abs(turn_degree_ISI))],
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
            trial_evaluation_list.append(trial_summary)
            trial_id = trial_id + 1
    raster_pd = pd.concat(raster_list)
    if "type" in df_summary.columns:
        raster_pd.columns = [
            "distance_from_centre",
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
            "distance_from_centre",
            "velocity",
            "omega",
            "normalised_v",
            "normalised_omega",
            "id",
            "mu",
            "density",
        ]
    dif_across_trials_pd = pd.concat(dif_across_trials)
    simulated_across_trials_pd = pd.concat(simulated_across_trials)
    if dif_across_trials_pd.shape[1] == 2:
        column_list = ["x", "y"]
    elif dif_across_trials_pd.shape[1] == 4:
        column_list = ["x", "y", "degree", "ts"]
    elif dif_across_trials_pd.shape[1] == 5:
        column_list = ["type", "x", "y", "degree", "ts"]
    elif dif_across_trials_pd.shape[1] == 6:
        column_list = ["type", "agent_id", "x", "y", "degree", "ts"]
    dif_across_trials_pd.columns=column_list
    simulated_across_trials_pd.columns=column_list

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
            sim_grp=simulated_across_trials_pd[(simulated_across_trials_pd['type']==keys[0])&(simulated_across_trials_pd['degree']==keys[1])]
            fig = plt.figure(figsize=(9,5))
            ax = fig.add_subplot(212)
            ax2 = fig.add_subplot(221)
            ax3 = fig.add_subplot(222)
            ax.hist(grp['ts'].values,bins=100,density=False,color='r')
            ax.hist(sim_grp['ts'].values,bins=100,density=False,color='tab:gray',alpha=0.3)
            #ax.set(xlim=(0,60),ylim=(0,0.05),title=f'agent:{keys[0]},deg:{int(keys[1])}')
            ax.set(xlim=(0,60),title=f'agent:{keys[0]},deg:{int(keys[1])}')
            if distribution_with_entire_body:
                body_points=generate_points_within_rectangles(grp['x'].values,grp['y'].values, 1,4,2,21)
                ax2.hist2d(body_points[:,0],body_points[:,1],bins=400)
                body_points=generate_points_within_rectangles(sim_grp['x'].values,sim_grp['y'].values, 1,4,2,21)
                ax3.hist2d(body_points[:,0],body_points[:,1],bins=400)
            else:
                ax2.hist2d(grp['x'].values,grp['y'].values,bins=100)
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

    agent_pattern = f"VR1*agent_full.h5"
    agent_file = find_file(Path(this_dir), agent_pattern)
    xy_pattern = f"VR1*XY_full.h5"
    focal_animal_file = find_file(Path(this_dir), xy_pattern)
    summary_pattern = f"VR1*score_full.h5"
    summary_file = find_file(Path(this_dir), summary_pattern)

    dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap,simulated_across_trials_pd = (
        calculate_relative_position(
            summary_file, focal_animal_file, agent_file, analysis_methods
        )
    )
    return dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap,simulated_across_trials_pd


if __name__ == "__main__":
    # thisDir = r"D:/MatrexVR_2024_Data/RunData/20241125_131510"
    #thisDir = r"D:/MatrexVR_grass1_Data/RunData/20240907_190839"
    thisDir = r"D:/MatrexVR_2024_Data/RunData/20241110_165438"
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241116_134457"
    #thisDir = r"D:/MatrexVR_2024_Data/RunData/20241225_134852"
    #thisDir =r"D:/MatrexVR_2024_Data/RunData/20241231_130927"
    # thisDir = r"D:/MatrexVR_2024_Data/RunData/20241201_131605"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
