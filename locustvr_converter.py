# this is a file to convert data from matrexVR to locustVR.
# Input: csv file, gz csv file from matrexVR
# output: h5 file that stores single animal's response in multiple conditions
import time
import pandas as pd
import numpy as np
import os, gzip, re, json, sys
from pathlib import Path
from threading import Lock
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

from deepdiff import DeepDiff
from pprint import pprint

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file
from data_cleaning import load_temperature_data
from funcs import removeNoiseVR, diskretize

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


def fill_missing_data(df):

    # # Iterate over the rows to identify missing data sequences in the 6th column
    # missing_sections = []
    # current_section = []
    # for i in range(len(df) - 1):
    #     if df.iloc[i, 5] == 0 and (df.iloc[i + 1, 4] == df.iloc[i, 4] + 1):
    #         current_section.append(i)
    #     elif current_section:
    #         # If we reached the end of a missing section, save it
    #         missing_sections.append(current_section)
    #         current_section = []

    # # Process each missing section
    # for section in missing_sections:
    #     # Find the first two normal values after the missing sequence in the 6th column
    #     start_idx = section[-1] + 1
    #     non_zero_values = df.iloc[start_idx:, 5].loc[lambda x: x != 0].iloc[:2]

    #     # Ensure there are at least two values for interpolation
    #     if len(non_zero_values) < 2:
    #         continue  # Skip if fewer than two values are found

    #     # Indices for interpolation reference points
    #     ref_idx1, ref_idx2 = non_zero_values.index[0], non_zero_values.index[1]
    #     ref_val1, ref_val2 = (
    #         df.at[ref_idx1, df.columns[5]],
    #         df.at[ref_idx2, df.columns[5]],
    #     )

    #     # Get values in the 12th column over the range for missing data rows
    #     col12_values = df.loc[section[0] : ref_idx1, df.columns[11]].values

    #     # Calculate the step for interpolation
    #     step = (ref_val2 - ref_val1) / (ref_idx2 - ref_idx1)

    #     # Fill in missing values using extrapolation logic
    #     for j, idx in enumerate(section):
    #         df.at[idx, df.columns[5]] = ref_val1 + step * (idx - ref_idx1)

    for i in range(len(df)):
        # Detect the start of a missing block where CurrentStep is 0 and CurrentTrial increments
        if "tmp" in locals() and "missing_end" in locals():
            if tmp.shape[0] < missing_end:
                skip_number = missing_end + tmp.shape[0]
            else:
                skip_number = tmp.shape[0]
            if i < skip_number:
                continue
            else:
                pass
        else:
            i = i
        # if df.loc[i, "GameObjectPosX"] == 0 and (
        #     i == 0 or df.loc[i, "CurrentStep"] == df.loc[i - 1, "CurrentStep"] + 1
        # ):# did not cover when changing trials
        if df.loc[i, "GameObjectPosX"] == 0:
            # Define the start and end of missing data block
            missing_start = i
            while i < len(df) and df.loc[i, "GameObjectPosX"] == 0:
                i += 1
            missing_end = i
            next_valid_steps = df.loc[
                missing_start : missing_end + 1, "GameObjectRotY"
            ].values
            next_valid_sens_pos = df.loc[
                missing_start : missing_end + 1, "SensRotY"
            ].values
            # if np.where(next_valid_sens_pos == 0)[0].shape[0] == 0:
            #     tmp = np.diff(np.unwrap(np.flip(next_valid_sens_pos), period=360))
            #     df.loc[missing_start + 1 : missing_end + 1, "GameObjectRotY"] = np.flip(
            #         np.unwrap(next_valid_steps[-1] + tmp, period=360)
            #     )
            # else:
            num_zero_fictrac = np.where(next_valid_sens_pos == 0)[0].shape[
                0
            ]  ## this check how many 0 in fictrac data
            tmp = np.diff(
                np.unwrap(np.flip(next_valid_sens_pos[num_zero_fictrac:]), period=360)
            )
            df.loc[
                missing_start + num_zero_fictrac + 1 : missing_end + 1,
                "GameObjectRotY",
            ] = np.flip(np.unwrap(next_valid_steps[-1] + tmp, period=360))
        else:
            continue

    return df


# # Apply the function to the data
# filled_data = fill_missing_data(data)


def ffill(arr):
    mask = np.isnan(arr)
    if arr.ndim == 1:
        Warning("work in progress")
        # idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        # np.maximum.accumulate(idx, out=idx)
        # out = arr[np.arange(idx.shape[0])[None], idx]
    elif arr.ndim == 2:
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


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


# Simple solution for bfill provided by financial_physician in comment below
def bfill(arr):
    if arr.ndim == 1:
        return ffill(arr[::-1])[::-1]
    elif arr.ndim == 2:
        return ffill(arr[:, ::-1])[:, ::-1]


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
        conditions = []
        result = []
        agent_pattern = "*_Choice_*.json"
        found_result = find_file(thisDir, agent_pattern)
        if found_result is None:
            return print(f"file with {agent_pattern} not found")
        else:
            condition_dict = {}
            if isinstance(found_result, list):
                print(
                    f"Analyze {agent_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )

                for this_file in found_result:
                    with open(this_file, "r") as f:
                        print(f"load analysis methods from file {this_file}")
                        condition_id = this_file.stem.split("_")[4]
                        tmp = json.loads(f.read())
                        condition = {
                            "type": tmp["objects"][0]["type"],
                            "radial_distance": tmp["objects"][0]["position"]["radius"],
                            "polar_angle": tmp["objects"][0]["position"]["angle"],
                            "mu": tmp["objects"][0]["mu"],
                            "speed": tmp["objects"][0]["speed"],
                        }
                        condition_dict[condition_id] = condition

            elif len(found_result.stem) > 0:
                with open(found_result, "r") as f:
                    print(f"load analysis methods from file {found_result}")
                    condition_id = found_result.stem.split("_")[4]
                    tmp = json.loads(f.read())
                    condition = {
                        "type": tmp["objects"][0]["type"],
                        "radial_distance": tmp["objects"][0]["position"]["radius"],
                        "polar_angle": tmp["objects"][0]["position"]["angle"],
                        "mu": tmp["objects"][0]["mu"],
                        "speed": tmp["objects"][0]["speed"],
                    }
        json_pattern = "*sequenceConfig.json"
        found_result = find_file(thisDir, json_pattern)
        with open(found_result, "r") as f:
            print(f"load conditions from file {found_result}")
            tmp = json.loads(f.read())
        for i in range(len(tmp["sequences"])):
            tmp["sequences"][i]["duration"]
            this_condition_file = (
                tmp["sequences"][i]["parameters"]["configFile"]
                .split("_")[1]
                .split(".")[0]
            )

            this_condition = condition_dict[this_condition_file]
            meta_condition = (this_condition, tmp["sequences"][i]["duration"])
            conditions.append(meta_condition)

        if len(df) > 0:
            v_phase = None
            for _, entries in df.groupby(["CurrentTrial", "CurrentStep"]):

                ct = pd.to_datetime(
                    entries["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
                )
                result.append(
                    pd.concat(
                        [ct, entries["GameObjectPosX"], entries["GameObjectPosZ"]],
                        axis=1,
                    )
                )  # need to add visibility phase here in the future
        else:
            result = [None, None, None, None]
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
    df_f = fill_missing_data(df)
    # grouped = df.groupby(["CurrentTrial", "CurrentStep"])
    # for name, entries in df_f.groupby(["CurrentTrial", "CurrentStep"]):
    #     # entries["dif_orientation"] = entries["SensRotY"].diff()
    #     # print(entries["dif_orientation"].cumsum().head(10))
    #     print(f'First 2 entries for the "{name}" category:')
    #     print(30 * "-")
    #     print(entries["SensPosX"].head(10), "\n\n")
    #     print(entries["GameObjectPosX"].head(10), "\n\n")
    # print(entries["SensRotY"].head(5), "\n\n")
    # test = df["GameObjectRotY"].values
    # num_zero_fictrac = np.where(test == 0)[0].shape[0]
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
        # aligned_THP = tem_df.reindex(df.index, method="nearest")
        # df = df.join(aligned_THP.astype(np.float32))
        # df = df.join(aligned_THP)
        del tem_df
    # if tem_df is not None:
    #     tem_df = tem_df.resample(f"{int(1000 / monitor_fps)}ms").interpolate()
    #     df.set_index("Current Time", inplace=True)
    #     df = df.join(tem_df.reindex(df.index, method="nearest").astype(np.float32))
    #     del tem_df

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
            df_simulated = df_simulated_animal[id // 2]
            df_simulated.set_index("Current Time", inplace=True)
            df_simulated = df_simulated.reindex(ts.index, method="nearest")
        elif (
            scene_name != "choice"
            and isinstance(df_simulated_animal[id], pd.DataFrame) == True
        ):
            these_simulated_agents = df_simulated_animal[id]
            these_simulated_agents = these_simulated_agents.reindex(
                ts.index, method="nearest"
            )  ## Has an error ValueError: cannot reindex on an axis with duplicate labels

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
            loss = np.nan
        else:
            ##need to think about whether applying removeNoiseVR only to spatial discretisation or general
            elapsed_time = None
            loss, X, Y = removeNoiseVR(xy[0], xy[1])
            loss = 1 - loss
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
            # [Optional] Step 2: Rotate counterclockwise by 90 degrees (add pi/2 radians)
            # angles_rotated = angles_rad + np.pi / 2
            # Step 3: Ensure the angles remain in the range [-π, π]
            angles = (angles_rad + np.pi) % (2 * np.pi) - np.pi
            temperature = df[this_range]["Temperature ˚C (ºC)"].values
            humidity = df[this_range]["Relative Humidity (%)"].values
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
        c = np.cos(angles)
        s = np.sin(angles)
        if len(angles) == 0:
            xm = ym = meanAngle = meanVector = VecSin = VecCos = np.nan
        else:
            xm = np.sum(c) / len(angles)
            ym = np.sum(s) / len(angles)
            meanAngle = np.arctan2(ym, xm)
            # ang_deg = np.rad2deg(ang_rad) ## if converting the unit to degree
            # ang_deg = np.mod(ang_deg,360.)# if the range is from 0 to 360
            meanVector = np.sqrt(np.square(np.sum(c)) + np.square(np.sum(s))) / len(
                angles
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
        elif isinstance(
            condition, tuple
        ):  # designed for the choice assay, which use tuple to carry duration
            duration = condition[1]  # drop the duration from the
            condition = condition[0]
        else:
            pass

        spe = [condition["speed"]] * len(dX)
        mu = [condition["mu"]] * len(dX)
        if scene_name.lower() == "swarm" or scene_name.lower() == "band":
            order = [condition["kappa"]] * len(dX)
            density = [condition["density"]] * len(dX)
            du = [condition["duration"]] * len(dX)
        if scene_name.lower() == "choice" or scene_name.lower() == "band":
            polar_angle = [condition["polar_angle"]] * len(dX)
            radial_distance = [condition["radial_distance"]] * len(dX)

        if scene_name.lower() == "choice":
            if condition["type"] == "LeaderLocust":
                object_type = ["mov_glocust"] * len(dX)
            elif condition["type"] == "":
                object_type = ["empty_trial"] * len(dX)
            du = [duration] * len(dX)
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
        elif agents_shared_across_vrs and "ts_simulated_animal" in locals():
            print(
                "Information about simulated locusts are shared across rigs in the choice scene, so skip the rest of the loop and start analysing focal animals"
            )
        else:

            df_simulated_animal = []
            conditions = []
            if isinstance(found_result, list):
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
                        num_object = len(trial_condition["objects"])
                    else:
                        trial_condition = trial_sequence["sequences"][idx]["parameters"]
                        num_object = 1
                    this_file = found_result[idx * num_object]
                    result_list = []
                    condition_list = []
                    for this_object in range(num_object):
                        if "objects" in trial_condition:
                            result, condition = read_agent_data(
                                found_result[idx * num_object + this_object],
                                analysis_methods,
                                trial_condition["objects"][this_object],
                            )
                        else:
                            result, condition = read_agent_data(
                                found_result[idx * num_object + this_object],
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
                    if num_object == 1:
                        conditions.append(condition_list[0])
                        df_simulated_animal.append(result_list[0])
                    else:
                        conditions.append(condition_list)
                        df_simulated_animal.append(result_list)

            elif len(found_result.stem) > 0:

                (
                    df_simulated_animal,
                    conditions,
                ) = read_agent_data(
                    found_result,
                    analysis_methods,
                )

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
    # thisDir = r"D:/MatrexVR_Swarm_Data/RunData/20240815_134157"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240816_145830"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240826_150826"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_171158"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_151537"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\archive\20240905_193855"
    thisDir = r"D:\MatrexVR_grass1_Data\RunData\20240907_142802"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
