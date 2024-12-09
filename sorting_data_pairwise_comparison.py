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


def calculate_speed(dif_x, dif_y, ts, number_frame_scene_changing=5):
    focal_distance_fbf = np.sqrt(np.sum([dif_x**2, dif_y**2], axis=0))
    focal_distance_fbf[0 : number_frame_scene_changing + 1] = (
        np.nan
    )  ##plus one to include the weird data from taking difference between 0 and some value
    instant_speed = focal_distance_fbf / np.diff(ts)
    return instant_speed


def time_series_plot(target_distance, instant_speed, angles, analysis_window):
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
    ax2.plot(np.arange(instant_speed.shape[0]), instant_speed)
    ax3.plot(np.arange(angles.shape[0]), angles)
    plt.show()


def behavioural_analysis(
    focal_xy, instant_speed, angular_velocity, epochs_of_interest, file_name, trial_id
):
    speed_threshold = 0.25
    walk_epochs = instant_speed > speed_threshold
    omega_threshold = 0.08
    turn_epochs = abs(angular_velocity) > omega_threshold
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4), tight_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.scatter(
        focal_xy[0, 1:],
        focal_xy[1, 1:],
        c="k",
        # c=np.zeros((1, focal_xy.shape[1] - 1), dtype=np.int8),
    )
    ax2.scatter(
        focal_xy[0, 1:],
        focal_xy[1, 1:],
        c=walk_epochs.astype(int),
        cmap="inferno",
    )
    ax3.scatter(
        focal_xy[0, 1:], focal_xy[1, 1:], c=turn_epochs.astype(int), cmap="inferno"
    )
    ax4.scatter(
        focal_xy[0, 1:],
        focal_xy[1, 1:],
        c=epochs_of_interest.astype(int),
        cmap="inferno",
    )
    fig_name = f"{file_name.stem.split('_')[0]}_{trial_id}_trajectory_analysis.jpg"
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
    # angular_velocity=np.diff(np.unwrap(angle_rad))
    ang_deg = np.mod(np.rad2deg(angle_rad), 360.0)  ## if converting the unit to degree
    angular_velocity = np.diff(
        np.unwrap(ang_deg, period=360)
    )  ##if converting the unit to degree
    angle_rad[0 : number_frame_scene_changing + 1] = (
        np.nan
    )  ##plus one to include the weird data from taking difference between 0 and some value
    angular_velocity[0 : number_frame_scene_changing + 1] = (
        np.nan
    )  ##plus one to include the weird data from taking difference between 0 and some value
    return angle_rad, angular_velocity


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
    # if analysis_methods.get("plotting_trajectory"):
    #     time_series_plot(target_distance,instant_speed,angles_in_degree,analysis_window)
    locustVR_criteria = np.logical_and(
        target_distance[1:] < follow_within_distance,
        instant_speed > 1,
        angles_in_degree < 10,
    )
    walk_criteria = np.logical_and(
        target_distance[1:] < follow_within_distance, instant_speed > 1
    )
    if extract_follow_epoches and follow_locustVR_criteria:
        epochs_of_interest = locustVR_criteria
    elif extract_follow_epoches:
        epochs_of_interest = walk_criteria
    else:
        epochs_of_interest = (
            np.ones((instant_speed.shape[0])) == 1.0
        )  # created a all-true array for overall heatmap
    return epochs_of_interest, vector_dif


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
    duration_for_baseline = 3
    analysis_window = analysis_methods.get("analysis_window")
    monitor_fps = analysis_methods.get("monitor_fps")
    align_with_isi_onset = analysis_methods.get("align_with_isi_onset", False)
    pre_stim_ISI = 60
    trajec_lim = 150
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
    if analysis_methods.get("plotting_trajectory"):
        plot_trajectory(df_focal_animal, df_summary, df_agent, focal_animal_file)
    dif_across_trials = []
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
        _, change_agular_degree_fbf = diff_angular_degree(
            heading_direction, num_unfilled_gap
        )
        angular_velocity = change_agular_degree_fbf / np.diff(ts)
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

        if grp["type"][0] == "empty_trial":
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
                for i in range(num_portion):
                    if i == 0:
                        this_agent_xy = agent_xy[:, :midpoint]  # First half
                        print(f"Processing first half: {this_agent_xy}")
                    else:
                        this_agent_xy = agent_xy[:, midpoint:]  # Second half
                        print(f"Processing second half: {this_agent_xy}")
                    epochs_of_interest, vector_dif = classify_follow_epochs(
                        focal_xy, instant_speed, ts, this_agent_xy, analysis_methods
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
                    follow_pd_list.append(follow_pd)
            else:
                epochs_of_interest, vector_dif = classify_follow_epochs(
                    focal_xy, instant_speed, ts, agent_xy, analysis_methods
                )
                behavioural_analysis(
                    focal_xy,
                    instant_speed,
                    angular_velocity,
                    epochs_of_interest,
                    focal_animal_file,
                    key,
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

            if "follow_pd_list" in locals():
                follow_pd_combined = pd.concat(follow_pd_list)
                dif_across_trials.append(follow_pd_combined)
                sum_follow_epochs = follow_pd_combined.shape[0]
            else:
                dif_across_trials.append(follow_pd)
                sum_follow_epochs = follow_pd.shape[0]
            _, turn_degree_fbf = diff_angular_degree(
                heading_direction, num_unfilled_gap
            )
            angular_velocity = turn_degree_fbf / np.diff(ts)
            trial_summary = pd.DataFrame(
                {
                    "trial_id": [trial_id],
                    "mu": [grp["mu"].values[0]],
                    "polar_angle": [grp["polar_angle"].values[0]],
                    # "this_vr": [grp['this_vr'][0]],
                    "num_follow_epochs": [sum_follow_epochs],
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
    if dif_across_trials_pd.shape[1] == 2:
        dif_across_trials_pd.columns = ["x", "y"]
    elif dif_across_trials_pd.shape[1] == 4:
        dif_across_trials_pd.columns = ["x", "y", "degree", "ts"]
    elif dif_across_trials_pd.shape[1] == 5:
        dif_across_trials_pd.columns = ["type", "x", "y", "degree", "ts"]
    elif dif_across_trials_pd.shape[1] == 6:
        dif_across_trials_pd.columns = ["type", "agent_id", "x", "y", "degree", "ts"]
    return dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap


def load_data(this_dir, json_file):
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

    dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap = (
        calculate_relative_position(
            summary_file, focal_animal_file, agent_file, analysis_methods
        )
    )
    return dif_across_trials_pd, trial_evaluation_list, raster_pd, num_unfilled_gap


if __name__ == "__main__":
    # thisDir = r"D:/MatrexVR_2024_Data/RunData/20241125_131510"
    thisDir = r"D:/MatrexVR_2024_Data/RunData/20241201_131605"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
