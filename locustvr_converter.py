# %%
# can not make plots during debug mode under the new virtual environment
# solution:  https://stackoverflow.com/questions/57160241/how-to-plot-during-debugger-mode-of-vscode
# this is a file to convert data from matrexVR to locustVR.
# Input: csv file, gz csv file from matrexVR
# output:
# h5 files storing single animal's response in the following forms
# information about single animal's position in every frame: curated_file = f"{experiment_id}_XY{file_suffix}.h5"
# summarise the response in every trials: summary_file = f"{experiment_id}_score{file_suffix}.h5"
# [Optional] information about agents: agent_file = f"{experiment_id}_agent{file_suffix}.h5"
#  Note: since the fictrac and Unity comes async, there is always one row delayed in Unity's GameObject dataset
# in terms of pos, those are filled with sv filter so that nan can not be detected anymore but it is visible in rotY
## Due to sometime RotY did not get fictrac data for some reasons for some rows, the rows with nan is 1 + (number of undetected rows) in the dataset.

import time
import pandas as pd
import numpy as np
import gzip, re, json, sys
from pathlib import Path
from threading import Lock
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

# from deepdiff import DeepDiff
# from pprint import pprint

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file, findLongestConseqSubseq
from data_cleaning import load_temperature_data, removeFictracNoise, bfill, diskretize

lock = Lock()


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


colormap_name = "coolwarm"
sm = cm.ScalarMappable(cmap=colormap_name)
COL = MplColorHelper(colormap_name, 0, 8)


def calcAffineMatrix(sourcePoints, targetPoints):
    # https://stackoverflow.com/questions/44674129/how-can-i-use-scipys-affine-transform-to-do-an-arbitrary-affine-transformation
    # For three or more source and target points, find the affine transformation
    A = []
    b = []
    for sp, trg in zip(sourcePoints, targetPoints):
        A.append([sp[0], 0, sp[1], 0, 1, 0])
        A.append([0, sp[0], 0, sp[1], 0, 1])
        b.append(trg[0])
        b.append(trg[1])
    result, resids, rank, s = np.linalg.lstsq(np.array(A), np.array(b))

    a0, a1, a2, a3, a4, a5 = result
    affineTrafo = np.float32([[a0, a2, a4], [a1, a3, a5]])
    return affineTrafo


def calculate_speed(dif_x, dif_y):
    focal_distance_fbf = np.sqrt(np.sum([dif_x**2, dif_y**2], axis=0))
    # focal_distance_fbf[0:number_frame_scene_changing+1]=np.nan##plus one to include the weird data from taking difference between 0 and some value
    # instant_speed=focal_distance_fbf/np.diff(ts)
    return focal_distance_fbf


def fill_missing_data(df):
    for i in range(len(df)):
        # Detect the start of a missing block where CurrentStep is 0 and CurrentTrial increments
        if "fill_rot" in locals() and "missing_end" in locals():
            if fill_rot.shape[0] < missing_end:
                skip_number = missing_end + fill_rot.shape[0]
            else:
                skip_number = fill_rot.shape[0]
            if i < skip_number:
                continue
            else:
                pass
        else:
            i = i
        if df.loc[i, "GameObjectPosX"] == 0:
            # Define the start and end of missing data block
            missing_start = i
            while i < len(df) and df.loc[i, "GameObjectPosX"] == 0:
                i += 1
            missing_end = i
            missing_heading = df.loc[
                missing_start : missing_end + 1, "GameObjectRotY"
            ].values
            reference_heading = df.loc[
                missing_start : missing_end + 1, "SensRotY"
            ].values
            reference_x = df.loc[missing_start : missing_end + 1, "SensPosX"].values
            reference_y = df.loc[missing_start : missing_end + 1, "SensPosY"].values

            go_x = df.loc[
                missing_start + 20 : missing_end + 20, "GameObjectPosX"
            ].values
            go_y = df.loc[
                missing_start + 20 : missing_end + 20, "GameObjectPosZ"
            ].values
            sens_x = df.loc[missing_start + 20 : missing_end + 20, "SensPosX"].values
            sens_y = df.loc[missing_start + 20 : missing_end + 20, "SensPosY"].values
            source_points = np.column_stack((sens_x, sens_y))
            target_points = np.column_stack((go_x, go_y))
            aff_m = calcAffineMatrix(source_points[:-1], target_points[1:])
            num = np.ones(reference_x.shape[0])
            reference_coordinates = np.column_stack((reference_x, reference_y, num))
            transformed_coordinates = reference_coordinates @ aff_m.T

            num_zero_fictrac = np.where(reference_heading == 0)[0].shape[
                0
            ]  ## this check how many 0 in fictrac data
            fill_rot = np.diff(
                np.unwrap(np.flip(reference_heading[num_zero_fictrac:]), period=360)
            )
            df.loc[
                missing_start + num_zero_fictrac + 1 : missing_end + 1,
                "GameObjectRotY",
            ] = np.flip(
                np.unwrap(missing_heading[-1] + np.cumsum(fill_rot), period=360)
            )
            df.loc[
                missing_start + num_zero_fictrac + 1 : missing_end + 1,
                "GameObjectPosX",
            ] = transformed_coordinates[num_zero_fictrac:-1, 0]
            df.loc[
                missing_start + num_zero_fictrac + 1 : missing_end + 1,
                "GameObjectPosZ",
            ] = transformed_coordinates[num_zero_fictrac:-1, 1]
            ##check if curation is correct
            # fig, axes = plt.subplots(2, 2, figsize=(18, 7), tight_layout=True)
            # ax, ax1, ax2, ax3 = axes.flatten()
            # reference_rot=df.loc[
            #     missing_start + num_zero_fictrac + 1 : missing_start + 30 + 1,
            #     "SensRotY",
            # ].values
            # curated_rot=df.loc[
            #     missing_start + num_zero_fictrac + 1 : missing_start + 30 + 1,
            #     "GameObjectRotY",
            # ].values
            # reference_heading_x = df.loc[
            #     missing_start + num_zero_fictrac + 1 : missing_start + 30 + 1,
            #     "SensPosX",
            # ].values
            # reference_heading_y = df.loc[
            #     missing_start + num_zero_fictrac + 1 : missing_start + 30 + 1,
            #     "SensPosY",
            # ].values
            # curated_heading_x = df.loc[
            #     missing_start + num_zero_fictrac + 1 : missing_start + 30 + 1,
            #     "GameObjectPosX",
            # ].values
            # curated_heading_y = df.loc[
            #     missing_start + num_zero_fictrac + 1 : missing_start + 30 + 1,
            #     "GameObjectPosZ",
            # ].values
            # x=np.arange(0,curated_rot.shape[0]-1)
            # ax.plot(x,np.diff(np.unwrap(reference_rot, period=360)))
            # ax.set(title="reference")
            # ax1.plot(x,np.diff(np.unwrap(curated_rot, period=360)))
            # ax1.set(title="curated")
            # dif_x = np.diff(reference_heading_x)
            # dif_y = np.diff(reference_heading_y)
            # diff_reference = calculate_speed(dif_x, dif_y)
            # dif_x = np.diff(curated_heading_x)
            # dif_y = np.diff(curated_heading_y)
            # diff_curated = calculate_speed(dif_x, dif_y)
            # ax2.plot(x, diff_reference * 5)
            # ax2.set(title="diff_reference")
            # ax3.plot(x, diff_curated)
            # ax3.set(title="diff_curated")
            # plt.show()
        else:
            continue

    return df


def reshape_multiagent_data(df, this_object):
    number_of_duplicates = df["Timestamp"].drop_duplicates().shape[0]
    number_of_instances = int(df.shape[0] / number_of_duplicates)
    agent_id = np.tile(
        np.arange(number_of_instances) + number_of_instances * this_object,
        number_of_duplicates,
    )
    c_name_list = ["agent" + str(num) for num in agent_id]
    test = pd.concat([df, pd.DataFrame(c_name_list)], axis=1)
    if "VisibilityPhase" in df.columns:
        df_values = ["X", "Z", "VisibilityPhase"]

    else:
        df_values = ["X", "Z"]
    new_df = test.pivot(index="Timestamp", columns=0, values=df_values)
    # new_df.loc[:, (slice(None), ["agent0"])] to access columns with multi-index
    return new_df


def prepare_data(df, this_range):
    ts = df.loc[this_range, "Current Time"]
    if ts.empty:
        return None
    x = df["GameObjectPosX"][this_range]
    y = df["GameObjectPosZ"][this_range]
    rot_y = df["GameObjectRotY"][this_range]
    xy = np.vstack((x.to_numpy(), y.to_numpy()))
    xy = bfill(xy)  # Fill missing values for smoother analysis
    trial_no = df.loc[this_range, "CurrentTrial"]
    return ts, xy, trial_no, rot_y


def load_file(file):
    if file.suffix == ".gz":
        with gzip.open(file, "rb") as f:
            return pd.read_csv(f)
    elif file.suffix == ".csv":
        return pd.read_csv(file)


def read_agent_data(this_file, analysis_methods, these_parameters=None):
    scene_name = analysis_methods.get("experiment_name")
    print("read simulated data")
    if type(this_file) == pd.DataFrame:
        df = this_file
    else:
        thisDir = this_file.parent
        if type(this_file) == str:
            this_file = Path(this_file)

        df = load_file(this_file)

    print(df.columns)
    if scene_name.lower() == "swarm":
        n_locusts = df.columns[6]
        boundary_size = df.columns[7]
        density = int(n_locusts.split(":")[1]) / (
            int(boundary_size.split(":")[1]) ** 2 / 10000
        )
        if these_parameters == None:
            mu = df.columns[8]
            kappa = df.columns[9]
            agent_speed = df.columns[10]
            conditions = {
                "density": float(density),
                mu.split(":")[0].lower(): int(mu.split(":")[1]),
                kappa.split(":")[0].lower(): float(kappa.split(":")[1]),
                "speed": float(agent_speed.split(":")[1]),
            }
        else:
            mu = these_parameters["mu"]
            kappa = these_parameters["kappa"]
            agent_speed = these_parameters["locustSpeed"]
            conditions = {
                "density": float(density),
                "mu": int(mu),
                "kappa": float(kappa),
                "speed": float(agent_speed),
            }
        # change the unit to m2

        if len(df) > 0:
            result = pd.concat(
                [
                    pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f"),
                    df["X"],
                    df["Z"],
                ],
                axis=1,
            )
        else:
            result = [
                pd.to_datetime(this_file.stem[0:19], format="%Y-%m-%d_%H-%M-%S"),
                None,
                None,
            ]

    elif scene_name.lower() == "band":
        conditions = {}
        for this_item in range(len(list(these_parameters))):
            if list(these_parameters)[this_item] == "position":
                conditions["radial_distance"] = these_parameters["position"]["radius"]
                conditions["polar_angle"] = these_parameters["position"]["angle"]
            else:
                conditions[list(these_parameters)[this_item]] = list(
                    these_parameters.values()
                )[this_item]
        conditions["density"] = conditions["numberOfInstances"] / (
            conditions["boundaryLengthX"] * conditions["boundaryLengthZ"] / 10000
        )
        if len(df) > 0:
            result = pd.concat(
                [
                    pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f"),
                    df["X"],
                    df["Z"],
                    df["VisibilityPhase"],
                ],
                axis=1,
            )
        else:
            result = [
                pd.to_datetime(this_file.stem[0:19], format="%Y-%m-%d_%H-%M-%S"),
                None,
                None,
                None,
            ]

    elif scene_name.lower() == "choice":
        conditions = {}
        for this_item in range(len(list(these_parameters))):
            if list(these_parameters)[this_item] == "position":
                conditions["radial_distance"] = these_parameters["position"]["radius"]
                conditions["polar_angle"] = these_parameters["position"]["angle"]
            else:
                conditions[list(these_parameters)[this_item]] = list(
                    these_parameters.values()
                )[this_item]
        if len(df) > 0:
            result = pd.concat(
                [
                    pd.to_datetime(df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"),
                    df["GameObjectPosX"],
                    df["GameObjectPosZ"],
                ],
                axis=1,
            )
        else:
            result = [None, None, None]
    return result, conditions


def analyse_focal_animal(
    this_file,
    analysis_methods,
    df_simulated_animal,
    conditions,
    tem_df=None,
):

    monitor_fps = analysis_methods.get("monitor_fps")
    plotting_trajectory = analysis_methods.get("plotting_trajectory", False)
    save_output = analysis_methods.get("save_output", False)
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset", False)
    time_series_analysis = analysis_methods.get("time_series_analysis", False)
    scene_name = analysis_methods.get("experiment_name")
    analyze_one_session_only = True
    BODY_LENGTH3 = (
        analysis_methods.get("body_length", 4) * 3
    )  ## multiple 3 because 3 body length is used for spatial discretisation in Sayin et al.
    growth_condition = analysis_methods.get("growth_condition")

    if type(this_file) == str:
        this_file = Path(this_file)

    df = load_file(this_file)
    df = fill_missing_data(df)
    test = np.where(df["GameObjectRotY"].values == 0)[0]
    longest_unity_gap = findLongestConseqSubseq(test, test.shape[0])
    print(f"longest unfilled gap is {longest_unity_gap}")
    # replace 0.0 with np.nan since they are generated during scene-switching
    ##if upgrading to pandas 3.0 in the future, try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead
    df.replace(
        {
            "GameObjectPosX": {0.0: np.nan},
            "GameObjectPosZ": {0.0: np.nan},
            "GameObjectRotY": {0.0: np.nan},
        },
        inplace=True,
    )
    df["Current Time"] = pd.to_datetime(
        df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    experiment_id = df["VR"][0] + " " + str(df["Current Time"][0]).split(".")[0]
    experiment_id = re.sub(r"\s+", "_", experiment_id)
    experiment_id = re.sub(r":", "", experiment_id)
    file_suffix = "_full" if time_series_analysis else ""
    curated_file_path = this_file.parent / f"{experiment_id}_XY{file_suffix}.h5"
    summary_file_path = this_file.parent / f"{experiment_id}_score{file_suffix}.h5"
    agent_file_path = this_file.parent / f"{experiment_id}_agent{file_suffix}.h5"
    # need to think about whether to name them the same regardless analysis methods

    if tem_df is None:
        df["Temperature ˚C (ºC)"] = np.nan
        df["Relative Humidity (%)"] = np.nan
    else:
        frequency_milisecond = int(1000 / monitor_fps)
        tem_df = tem_df.resample(
            f"{frequency_milisecond}L"
        ).interpolate()  # FutureWarning: 'L' is deprecated and will be removed in a future version, please use 'ms' instead.
        df.set_index("Current Time", drop=False, inplace=True)
        df = df.join(tem_df.reindex(df.index, method="nearest"))
        del tem_df

    if overwrite_curated_dataset and summary_file_path.is_file():
        summary_file_path.unlink(missing_ok=True)
        curated_file_path.unlink(missing_ok=True)
        agent_file_path.unlink(missing_ok=True)

    if plotting_trajectory:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), tight_layout=True)
        ax1.set_title("ISI")
        ax2.set_title("Trial")
    (
        heading_direction_across_trials,
        x_across_trials,
        y_across_trials,
        ts_across_trials,
    ) = ([], [], [], [])

    for id, condition in enumerate(conditions):
        this_range = (df["CurrentStep"] == id) & (df["CurrentTrial"] == 0)
        ts, xy, trial_no, rot_y = prepare_data(df, this_range)
        if len(ts) == 0:
            break
        elif len(trial_no.value_counts()) > 1 & analyze_one_session_only == True:
            break
        fchop = ts.iloc[0].strftime("%Y-%m-%d_%H%M%S")
        if scene_name == "choice" and id % 2 > 0:
            df_simulated = df_simulated_animal[id]
            if "Current Time" in df_simulated.columns:
                df_simulated.set_index("Current Time", inplace=True)
            df_simulated = df_simulated.reindex(ts.index, method="nearest")
        elif (
            scene_name != "choice"
            and isinstance(df_simulated_animal[id], pd.DataFrame) == True
        ):
            these_simulated_agents = df_simulated_animal[id]
            these_simulated_agents = these_simulated_agents.reindex(
                ts.index, method="nearest"
            )

        if time_series_analysis:
            elapsed_time = (ts - ts.min()).dt.total_seconds().values
            if analysis_methods.get("filtering_method") == "sg_filter":
                X = savgol_filter(xy[0], 59, 3, axis=0)
                Y = savgol_filter(xy[1], 59, 3, axis=0)
            else:
                X = xy[0]
                Y = xy[1]
            angles_rad = np.radians(
                -rot_y.values
            )  # turn negative to acount for Unity's axis and turn radian
            #if id == 119 and experiment_id=='VR4_2024-11-16_155242':
            remains, X, Y = removeFictracNoise(X, Y, analysis_methods)
            loss = 1 - remains
        else:
            ##need to think about whether applying removeNoiseVR only to spatial discretisation or general
            elapsed_time = None
            remains, X, Y = removeFictracNoise(xy[0], xy[1], analysis_methods)
            loss = 1 - remains
            if len(X) == 0:
                print("all is noise")
                continue

        theta = np.radians(-90)  # applying rotation matrix to rotate the coordinates
        # includes a minus because the radian circle is clockwise in Unity, so 45 degree should be used as -45 degree in the regular radian circle
        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rXY = rot_matrix @ np.vstack((X, Y))

        if time_series_analysis:
            (dX, dY) = (rXY[0], rXY[1])
            # Ensure the angles remain in the range [-π, π]
            angles = (angles_rad + np.pi) % (2 * np.pi) - np.pi
            temperature = df[this_range]["Temperature ˚C (ºC)"].values
            humidity = df[this_range]["Relative Humidity (%)"].values
            num_spatial_decision = len(angles)
        else:
            newindex = diskretize(list(rXY[0]), list(rXY[1]), BODY_LENGTH3)
            dX = rXY[0][newindex]
            dY = rXY[1][newindex]
            temperature = df.iloc[newindex]["Temperature ˚C (ºC)"].values
            humidity = df.iloc[newindex]["Relative Humidity (%)"].values
            angles = np.arctan2(np.diff(dY), np.diff(dX))
            angles = np.insert(
                angles, 0, np.nan
            )  # add the initial heading direction, which is an nan to avoid bias toward certain degree.
            num_spatial_decision = len(angles) - 1
        c = np.cos(angles)
        s = np.sin(angles)
        if len(angles) == 0:
            xm = ym = meanAngle = meanVector = VecSin = VecCos = np.nan
        else:
            xm = np.nansum(c) / num_spatial_decision
            ym = np.nansum(s) / num_spatial_decision
            meanAngle = np.arctan2(ym, xm)
            # ang_deg = np.rad2deg(ang_rad) ## if converting the unit to degree
            # ang_deg = np.mod(ang_deg,360.)# if the range is from 0 to 360
            meanVector = (
                np.sqrt(np.square(np.nansum(c)) + np.square(np.nansum(s)))
                / num_spatial_decision
            )
            VecSin = meanVector * np.sin(meanAngle)
            VecCos = meanVector * np.cos(meanAngle)
        std = np.sqrt(2 * (1 - meanVector))

        tdist = (
            np.sum(np.sqrt(np.add(np.square(np.diff(X)), np.square(np.diff(Y)))))
            if time_series_analysis
            else len(dX) * BODY_LENGTH3
        )  ##The distance calculated based on spatial discretisation should be the shortest

        f = [fchop] * len(dX)
        different_key = None
        if isinstance(condition, list):
            diff_con = DeepDiff(condition[0], condition[1])
            pprint(diff_con)
            different_key = diff_con.affected_root_keys[0]
            condition = condition[0]

        #   condition_is_tuple = False
        # elif isinstance(
        #     condition, tuple
        # ):  # designed for the choice assay, which use tuple to carry duration
        #     condition_is_tuple = True
        #     duration = condition[1]  # drop the duration from the
        #     condition = condition[0]
        # else:
        #     condition_is_tuple = False
        #     pass

        spe = [condition["speed"]] * len(dX)
        mu = [condition["mu"]] * len(dX)
        du = [condition["duration"]] * len(dX)
        if scene_name.lower() == "swarm" or scene_name.lower() == "band":
            order = [condition["kappa"]] * len(dX)
            density = [condition["density"]] * len(dX)
        if scene_name.lower() == "choice" or scene_name.lower() == "band":
            polar_angle = [condition["polar_angle"]] * len(dX)
            radial_distance = [condition["radial_distance"]] * len(dX)

        if scene_name.lower() == "choice":
            if condition["type"] == "LeaderLocust":
                object_type = ["mov_glocust"] * len(dX)
            elif condition["type"] == "LeaderLocust_black":
                object_type = ["mov_locustb"] * len(dX)
            elif condition["type"] == "InanimatedLeaderLocust_black":
                object_type = ["sta_locustb"] * len(dX)
            elif condition["type"] == "":
                object_type = ["empty_trial"] * len(dX)

        if scene_name.lower() == "band":
            voff = [condition["visibleOffDuration"]] * len(dX)
            von = [condition["visibleOnDuration"]] * len(dX)
            f_angle = [condition["rotationAngle"]] * len(dX)
            object_type = [condition["type"]] * len(dX)

        groups = [growth_condition] * len(dX)
        df_curated = pd.DataFrame(
            {
                "X": dX,
                "Y": dY,
                "heading": angles,
                "fname": f,
                "mu": mu,
                "agent_speed": spe,
                "duration": du,
            }
        )
        if type(elapsed_time) == np.ndarray:
            df_curated["ts"] = list(elapsed_time)
        if "temperature" in locals():
            df_curated["temperature"] = list(temperature)
            df_curated["humidity"] = list(humidity)
        if scene_name.lower() == "swarm" or scene_name.lower() == "band":
            df_curated["density"] = density
            df_curated["kappa"] = order
        if scene_name.lower() == "choice" or scene_name.lower() == "band":
            df_curated["type"] = object_type
            df_curated["radial_distance"] = radial_distance
            df_curated["polar_angle"] = polar_angle
            # Probably no need to save the following into curated database but just in case
            # if scene_name.lower() == "band":
            #     df_curated["visibleOffDuration"] = voff
            #     df_curated["visibleOnDuration"] = von
            #     df_curated["rotationAngle"] = f_angle

            ##load information about simulated locusts
            if scene_name.lower() == "choice":
                if (
                    id % 2 == 0
                ):  # stimulus trial is always the odd trial under choice scene
                    pass
                    # print("no information about ISI stored in choice assay")
                else:
                    agent_xy = np.vstack(
                        (
                            df_simulated["GameObjectPosX"].values,
                            df_simulated["GameObjectPosZ"].values,
                        )
                    )
                    agent_rXY = rot_matrix @ np.vstack((agent_xy[0], agent_xy[1]))
                    if time_series_analysis:
                        (agent_dX, agent_dY) = (agent_rXY[0], agent_rXY[1])
                    else:
                        agent_dX = agent_rXY[0][newindex]
                        agent_dY = agent_rXY[1][newindex]

            elif (
                isinstance(df_simulated_animal[id], pd.DataFrame) == True
                and "these_simulated_agents" in locals()
            ):
                Warning("work in progress")
                pass
            if "agent_dX" in locals():
                df_agent = pd.DataFrame(
                    {
                        "X": agent_dX,
                        "Y": agent_dY,
                        "fname": [fchop] * len(agent_dX),
                        "mu": mu,
                        "agent_speed": spe,
                    }
                )
                if scene_name.lower() == "swarm":
                    print(
                        "there is a unfixed bug about how to name the number of agent"
                    )
                    df_agent["agent_num"] = density
                elif scene_name.lower() == "choice":
                    df_agent["agent_no"] = [0] * len(
                        agent_dX
                    )  # need to figure out a way to solve multiple agents situation. The same method should be applied in the Swarm scene
                else:
                    pass

        df_summary = pd.DataFrame(
            {
                "fname": [f[0]],
                "loss": [loss],
                "mu": [mu[0]],
                "speed": [spe[0]],
                "groups": [groups[0]],
                "mean_angle": [meanAngle],
                "vector": [meanVector],
                "variance": [std],
                "distX": [dX[-1]],
                "distTotal": [tdist],
                "sin": [VecSin],
                "cos": [VecCos],
                "duration": [du[0]],
            }
        )
        if scene_name.lower() == "swarm" or scene_name.lower() == "band":
            df_summary["density"] = [density[0]]
            df_summary["kappa"] = [order[0]]
        if scene_name.lower() == "choice" or scene_name.lower() == "band":
            df_summary["type"] = [object_type[0]]
            df_summary["radial_distance"] = [radial_distance[0]]
            df_summary["polar_angle"] = [polar_angle[0]]
        if scene_name.lower() == "band":
            df_summary["visibleOffDuration"] = [voff[0]]
            df_summary["visibleOnDuration"] = [von[0]]
            df_summary["rotationAngle"] = [f_angle[0]]

        if plotting_trajectory == True:
            if scene_name.lower() == "swarm" or scene_name.lower() == "band":
                if df_summary["density"][0] > 0:
                    ## if using plot instead of scatter plot
                    ax2.plot(dX, dY)
                    ##blue is earlier colour and yellow is later colour
                else:
                    ax1.plot(dX, dY)
            elif scene_name.lower() == "choice":
                if df_summary["type"][0] == "empty_trial":
                    ax1.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                    )
                else:
                    ax2.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                    )
                    if "agent_dX" in locals():
                        ax2.plot(
                            agent_dX,
                            agent_dY,
                            c="k",
                            linewidth=1,
                        )

        #######################Sections to save data
        if save_output == True:
            with lock:
                if "df_agent" in locals():
                    file_list = [curated_file_path, summary_file_path, agent_file_path]
                    data_frame_list = [df_curated, df_summary, df_agent]
                else:
                    file_list = [curated_file_path, summary_file_path]
                    data_frame_list = [df_curated, df_summary]
                for this_name, this_pd in zip(file_list, data_frame_list):
                    store = pd.HDFStore(this_name)
                    if different_key != None and different_key in this_pd.columns:
                        this_pd[different_key] = np.nan
                    store.append(
                        "name_of_frame",
                        this_pd,
                        format="t",
                        data_columns=this_pd.columns,
                    )
                    store.close()

        heading_direction_across_trials.append(angles)
        x_across_trials.append(dX)
        y_across_trials.append(dY)
        if time_series_analysis:
            ts_across_trials.append(elapsed_time)
        else:
            ts_across_trials.append(ts)
        if "agent_dX" in locals():
            del agent_dX, agent_dY, df_agent
    trajectory_fig_path = this_file.parent / f"{experiment_id}_trajectory.png"
    if plotting_trajectory == True and save_output == True:
        fig.savefig(trajectory_fig_path)
    return (
        heading_direction_across_trials,
        x_across_trials,
        y_across_trials,
        ts_across_trials,
    )


def preprocess_matrex_data(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())

    found_result = find_file(thisDir, "matrexVR*.txt", "DL220THP*.csv")
    ## here to load temperature data
    if found_result is None:
        tem_df = None
        print(f"temperature file not found")

    else:
        if isinstance(found_result, list):
            print(
                f"Multiple temperature files are detected. Have not figured out how to deal with this."
            )
            for this_file in found_result:
                tem_df = load_temperature_data(this_file)
        else:
            tem_df = load_temperature_data(found_result)
        if (
            "Celsius(°C)" in tem_df.columns
        ):  # make the column name consistent with data from DL220 logger
            tem_df.rename(
                columns={
                    "Celsius(°C)": "Temperature ˚C (ºC)",
                    "Humidity(%rh)": "Relative Humidity (%)",
                },
                inplace=True,
            )
    num_vr = 4
    agents_shared_across_vrs = analysis_methods.get("agents_shared_across_vrs", False)
    scene_name = analysis_methods.get("experiment_name")
    ## here to load simulated agent's data
    for i in range(num_vr):

        if scene_name.lower() == "swarm" or scene_name.lower() == "band":
            agent_pattern = f"*SimulatedLocustsVR{i+1}*"
        elif scene_name.lower() == "choice":
            agent_pattern = "*Leader*"
        found_result = find_file(thisDir, agent_pattern)
        if found_result is None:
            return print(f"file with {agent_pattern} not found")
        elif agents_shared_across_vrs and "df_simulated_animal" in locals():
            print(
                "Information about simulated locusts are shared across rigs in the choice scene, so skip the rest of the loop and start analysing focal animals"
            )
        else:

            df_simulated_animal = []
            conditions = []
            if isinstance(found_result, list):
                num_type_object = len(found_result)
                print(
                    f"Analyze {agent_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )
                json_pattern = "*sequenceConfig.json"
                json_file = find_file(thisDir, json_pattern)
                with open(json_file, "r") as f:
                    print(f"load trial sequence from file {json_file}")
                    trial_sequence = json.loads(f.read())
                for idx in range(len(trial_sequence["sequences"])):
                    if "configFile" in trial_sequence["sequences"][idx]["parameters"]:
                        config_file = trial_sequence["sequences"][idx]["parameters"][
                            "configFile"
                        ]  # need to figure out how to deal with swarm data
                        this_config_file = find_file(thisDir, "*" + config_file)
                        with open(this_config_file, "r") as f:
                            print(f"load trial conditions from file {this_config_file}")
                            trial_condition = json.loads(f.read())
                        num_object_on_scene = len(trial_condition["objects"])
                    else:
                        trial_condition = trial_sequence["sequences"][idx]["parameters"]
                        num_object_on_scene = 1
                    # this_file = found_result[idx * num_object_on_scene]
                    result_list = []
                    condition_list = []
                    if scene_name.lower() == "choice":
                        df_list = []
                        if "df" not in locals():
                            for j in range(len(found_result)):
                                df_list.append(load_file(found_result[j]))
                            df = pd.concat(df_list, ignore_index=True)
                        this_range = (df["CurrentStep"] == idx) & (
                            df["CurrentTrial"] == 0
                        )
                        result, condition = read_agent_data(
                            df[this_range],
                            analysis_methods,
                            trial_condition["objects"][0],
                        )
                        result_list.append(result)
                    else:
                        for this_object in range(num_object_on_scene):
                            if "objects" in trial_condition:
                                result, condition = read_agent_data(
                                    found_result[
                                        idx * num_object_on_scene + this_object
                                    ],
                                    analysis_methods,
                                    trial_condition["objects"][this_object],
                                )
                            else:
                                result, condition = read_agent_data(
                                    found_result[
                                        idx * num_object_on_scene + this_object
                                    ],
                                    analysis_methods,
                                    trial_condition,
                                )
                            if isinstance(result, pd.DataFrame) == True:
                                df_agent = reshape_multiagent_data(result, this_object)
                            else:
                                df_agent = None
                            result_list.append(df_agent)
                    condition["duration"] = trial_sequence["sequences"][idx][
                        "duration"
                    ]  # may need to add condition to exclude some kind of data from choice assay.
                    condition_list.append(condition)
                    if num_object_on_scene == 1:
                        conditions.append(condition_list[0])
                        df_simulated_animal.append(result_list[0])
                    else:
                        conditions.append(condition_list)
                        df_simulated_animal.append(result_list)

            elif len(found_result.stem) > 0:
                json_pattern = "*sequenceConfig.json"
                json_file = find_file(thisDir, json_pattern)
                with open(json_file, "r") as f:
                    print(f"load trial sequence from file {json_file}")
                    trial_sequence = json.loads(f.read())
                for idx in range(len(trial_sequence["sequences"])):
                    if "configFile" in trial_sequence["sequences"][idx]["parameters"]:
                        config_file = trial_sequence["sequences"][idx]["parameters"][
                            "configFile"
                        ]  # need to figure out how to deal with swarm data
                        this_config_file = find_file(thisDir, "*" + config_file)
                        with open(this_config_file, "r") as f:
                            print(f"load trial conditions from file {this_config_file}")
                            trial_condition = json.loads(f.read())
                        num_object_on_scene = len(trial_condition["objects"])
                    else:
                        trial_condition = trial_sequence["sequences"][idx]["parameters"]
                        num_object_on_scene = 1
                    result_list = []
                    condition_list = []
                    if scene_name.lower() == "choice":
                        df = load_file(found_result)
                        this_range = (df["CurrentStep"] == idx) & (
                            df["CurrentTrial"] == 0
                        )
                        result, condition = read_agent_data(
                            df[this_range],
                            analysis_methods,
                            trial_condition["objects"][0],
                        )
                        result_list.append(result)

                        condition["duration"] = trial_sequence["sequences"][idx][
                            "duration"
                        ]  # may need to add condition to exclude some kind of data from choice assay.
                        condition_list.append(condition)
                        if num_object_on_scene == 1:
                            conditions.append(condition_list[0])
                            df_simulated_animal.append(result_list[0])
                        else:
                            conditions.append(condition_list)
                            df_simulated_animal.append(result_list)

        ## here to load focal_animal's data
        animal_name_pattern = f"*_VR{i+1}*"
        found_result = find_file(thisDir, animal_name_pattern)
        if found_result is None:
            return print(f"file with {animal_name_pattern} not found")
        else:
            if isinstance(found_result, list):
                print(
                    f"Analyze {animal_name_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )
                for this_file in found_result:
                    (
                        _,
                        _,
                        _,
                        _,
                    ) = analyse_focal_animal(
                        this_file,
                        analysis_methods,
                        df_simulated_animal,
                        conditions,
                        tem_df,
                    )
            elif len(found_result.stem) > 0:
                (
                    _,
                    _,
                    _,
                    _,
                ) = analyse_focal_animal(
                    found_result,
                    analysis_methods,
                    df_simulated_animal,
                    conditions,
                    tem_df,
                )


if __name__ == "__main__":
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240818_170807"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240824_143943"
    # thisDir = r"D:\MatrexVR_navigation_Data\RunData\20241012_162147"
    # thisDir = r"D:\MatrexVR_navigation_Data\RunData\archive\20241014_194555"
    # thisDir = r"D:/MatrexVR_Swarm_Data/RunData/20240815_134157"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240816_145830"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240826_150826"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_171158"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_151537"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\archive\20240905_193855"
    # thisDir = r"D:\MatrexVR_grass1_Data\RunData\20240907_142802"
    # thisDir = r"D:\MatrexVR_2024_Data\RunData\20241112_150308"
    thisDir = r"D:\MatrexVR_2024_Data\RunData\20241116_155210"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
