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


# Simple solution for bfill provided by financial_physician in comment below
def bfill(arr):
    if arr.ndim == 1:
        return ffill(arr[::-1])[::-1]
    elif arr.ndim == 2:
        return ffill(arr[:, ::-1])[:, ::-1]


def read_simulated_data(this_file, analysis_methods):
    scene_name = analysis_methods.get("experiment_name")

    print("read simulated data")
    if type(this_file) == str:
        this_file = Path(this_file)
    thisDir = this_file.parent
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    print(df.columns)
    if scene_name.lower() == "swarm":
        n_locusts = df.columns[6]
        boundary_size = df.columns[7]
        mu = df.columns[8]
        kappa = df.columns[9]
        agent_speed = df.columns[10]
        density = int(n_locusts.split(":")[1]) / (
            int(boundary_size.split(":")[1]) ** 2 / 10000
        )  # change the unit to m2
        conditions = {
            "Density": density,
            mu.split(":")[0]: int(mu.split(":")[1]),
            kappa.split(":")[0]: float(kappa.split(":")[1]),
            agent_speed.split(":")[0]: float(agent_speed.split(":")[1]),
        }
        if len(df) > 0:
            ts = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
            x = df["X"]
            y = df["Z"]
        else:
            ts = pd.to_datetime(this_file.stem[0:19], format="%Y-%m-%d_%H-%M-%S")
            x = None
            y = None
    elif scene_name.lower() == "choice":
        conditions = []
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
                            "agent": tmp["objects"][0]["type"],
                            "distance": tmp["objects"][0]["position"]["radius"],
                            "heading_angle": tmp["objects"][0]["position"]["angle"],
                            "walking_direction": tmp["objects"][0]["mu"],
                            "agent_speed": tmp["objects"][0]["speed"],
                        }
                        condition_dict[condition_id] = condition
                    # conditions.append(condition)

            elif len(found_result.stem) > 0:
                with open(found_result, "r") as f:
                    print(f"load analysis methods from file {found_result}")
                    condition_id = found_result.stem.split("_")[4]
                    tmp = json.loads(f.read())
                    condition = {
                        "agent": tmp["objects"][0]["type"],
                        "distance": tmp["objects"][0]["position"]["radius"],
                        "heading_angle": tmp["objects"][0]["position"]["angle"],
                        "walking_direction": tmp["objects"][0]["mu"],
                        "agent_speed": tmp["objects"][0]["speed"],
                    }
                    condition_dict[condition_id] = condition

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
            # if (
            #     i == 0
            # ):  ## need to add this condition because I hardcode to make the first empty scene 240 sec
            #     meta_condition = (this_condition, 240)
            # else:
            meta_condition = (this_condition, tmp["sequences"][i]["duration"])
            conditions.append(meta_condition)

        if len(df) > 0:
            ts = []
            x = []
            y = []
            for _, entries in df.groupby(["CurrentTrial", "CurrentStep"]):
                ts.append(
                    pd.to_datetime(
                        entries["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
                    )
                )
                x.append(entries["GameObjectPosX"])
                y.append(entries["GameObjectPosZ"])
        else:
            ts = None
            x = None
            y = None
    return ts, x, y, conditions


def prepare_data(df, this_range):
    ts = df.loc[this_range, "Current Time"]
    if ts.empty:
        return None
    # x, y = df.loc[this_range, ["GameObjectPosX", "GameObjectPosZ"]].T
    x = df["GameObjectPosX"][this_range]
    y = df["GameObjectPosZ"][this_range]

    xy = np.vstack((x.to_numpy(), y.to_numpy()))
    xy = bfill(xy)  # Fill missing values for smoother analysis
    trial_no = df.loc[this_range, "CurrentTrial"]
    return ts, xy, trial_no


def load_file(file):
    if file.suffix == ".gz":
        with gzip.open(file, "rb") as f:
            return pd.read_csv(f)
    elif file.suffix == ".csv":
        return pd.read_csv(file)


def analyse_focal_animal(
    this_file,
    analysis_methods,
    ts_simulated_animal,
    x_simulated_animal,
    y_simulated_animal,
    conditions,
    tem_df=None,
):

    monitor_fps = analysis_methods.get("monitor_fps")
    # camera_fps = analysis_methods.get("camera_fps")
    plotting_trajectory = analysis_methods.get("plotting_trajectory", False)
    save_output = analysis_methods.get("save_output", False)
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset", False)
    time_series_analysis = analysis_methods.get("time_series_analysis", False)
    scene_name = analysis_methods.get("experiment_name")
    alpha_dictionary = {0.1: 0.2, 1.0: 0.4, 10.0: 0.6, 100000.0: 1}
    analyze_one_session_only = True
    BODY_LENGTH3 = (
        analysis_methods.get("body_length", 4) * 3
    )  ## multiple 3 because 3 body length is used for spatial discretisation in Sayin et al.
    growth_condition = analysis_methods.get("growth_condition")

    this_file = Path(this_file) if isinstance(this_file, str) else this_file
    df = load_file(this_file)
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
    experiment_id = re.sub(
        r"\s+|:",
        "_",
        f'{df["VR"][0]} {df["Current Time"][0].strftime("%Y-%m-%d %H:%M:%S")}',
    )
    # experiment_id1 = df["VR"][0] + " " + str(df["Current Time"][0]).split(".")[0]
    # experiment_id1 = re.sub(r"\s+", "_", experiment_id)
    # experiment_id1 = re.sub(r":", "", experiment_id)
    file_suffix = "full" if time_series_analysis else ""
    curated_file_path = this_file.parent / f"{experiment_id}_XY_{file_suffix}.h5"
    summary_file_path = this_file.parent / f"{experiment_id}_score_{file_suffix}.h5"
    agent_file_path = this_file.parent / f"{experiment_id}_agent_{file_suffix}.h5"
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
        aligned_THP = tem_df.reindex(df.index, method="nearest")
        df = df.join(aligned_THP.astype(np.float32))
        # df = df.join(aligned_THP)
        del tem_df
    # if tem_df is not None:
    #     tem_df = tem_df.resample(f"{int(1000 / monitor_fps)}ms").interpolate()
    #     df.set_index("Current Time", inplace=True)
    #     df = df.join(tem_df.reindex(df.index, method="nearest").astype(np.float32))
    #     del tem_df
    # else:
    #     df["Temperature ˚C (ºC)"] = np.nan
    #     df["Relative Humidity (%)"] = np.nan

    if overwrite_curated_dataset and summary_file_path.is_file():
        summary_file_path.unlink(missing_ok=True)
        curated_file_path.unlink(missing_ok=True)
        agent_file_path.unlink(missing_ok=True)

    # if overwrite_curated_dataset == True and summary_file_path.is_file():
    #     summary_file_path.unlink()
    #     try:
    #         curated_file_path.unlink()
    #         agent_file_path.unlink()
    #     except OSError as e:
    #         # If it fails, inform the user.
    #         print("Error: %s - %s." % (e.filename, e.strerror))

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

    # for id in range(len(conditions)):
    for id, condition in enumerate(conditions):
        this_range = (df["CurrentStep"] == id) & (df["CurrentTrial"] == 0)
        # ts = df["Current Time"][this_range].astype("datetime64[ms]")
        # ts = df["Current Time"][this_range]
        data = prepare_data(df, this_range)
        if data is None:
            break
        ts, xy, trial_no = data
        fchop = ts.iloc[0].strftime("%Y-%m-%d_%H%M%S")
        if len(trial_no.value_counts()) > 1 & analyze_one_session_only == True:
            break
        # heading_direction = df["GameObjectRotY"][this_range]
        # x = df["GameObjectPosX"][this_range].astype(np.float32)
        # y = df["GameObjectPosZ"][this_range].astype(np.float32)
        # x = df["GameObjectPosX"][this_range]
        # y = df["GameObjectPosZ"][this_range]
        # xy = np.vstack((x.to_numpy(), y.to_numpy()))
        # # since I introduced nan earlier for the switch scene, I need to fill them with some values otherwise, smoothing methods will fail
        # xy = bfill(xy)
        # # trial_no = df["CurrentTrial"][this_range].astype("int16")
        # trial_no = df["CurrentTrial"][this_range]
        if scene_name == "choice" and id % 2 > 0:
            df_simulated = pd.concat(
                [
                    ts_simulated_animal[id // 2],
                    x_simulated_animal[id // 2],
                    y_simulated_animal[id // 2],
                ],
                axis=1,
            )
            df_simulated.set_index("Current Time", inplace=True)
            df_simulated = df_simulated.reindex(ts.index, method="nearest")

        elapsed_time = (
            (ts - ts.min()).dt.total_seconds().values if time_series_analysis else None
        )

        if time_series_analysis:
            if analysis_methods.get("filtering_method") == "sg_filter":
                X = savgol_filter(xy[0], 59, 3, axis=0)
                Y = savgol_filter(xy[1], 59, 3, axis=0)
            else:
                X = xy[0]
                Y = xy[1]
            loss = np.nan
        else:
            ##need to think about whether applying removeNoiseVR only to spatial discretisation or general
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

        # Calculate mean angles, vector, etc.
        angles = np.arctan2(np.diff(rXY[1]), np.diff(rXY[0]))

        if time_series_analysis:
            (dX, dY) = (rXY[0], rXY[1])
            temperature = df[this_range]["Temperature ˚C (ºC)"].values
            humidity = df[this_range]["Relative Humidity (%)"].values
        else:
            newindex = diskretize(list(rXY[0]), list(rXY[1]), BODY_LENGTH3)
            dX = rXY[0][newindex]
            dY = rXY[1][newindex]
            temperature = df.iloc[newindex]["Temperature ˚C (ºC)"].values
            humidity = df.iloc[newindex]["Relative Humidity (%)"].values

        c = np.cos(angles)
        s = np.sin(angles)
        if len(angles) == 0:
            xm = ym = meanAngle = meanVector = VecSin = VecCos = np.nan
        else:
            xm = np.sum(c) / len(angles)
            ym = np.sum(s) / len(angles)

            meanAngle = np.arctan2(ym, xm)
            meanVector = np.sqrt(np.square(np.sum(c)) + np.square(np.sum(s))) / len(
                angles
            )
            # sin = meanVector * np.sin(meanAngle)
            # cos = meanVector * np.cos(meanAngle)
            VecSin = meanVector * np.sin(meanAngle, dtype=np.float32)
            VecCos = meanVector * np.cos(meanAngle, dtype=np.float32)
        std = np.sqrt(2 * (1 - meanVector))

        tdist = (
            np.sum(np.sqrt(np.add(np.diff(X) ** 2, np.diff(Y) ** 2)))
            if time_series_analysis
            else len(X) * BODY_LENGTH3
        )  ##note this distance can be a lot larger than calculating with spatial discretisation

        f = [fchop] * len(dX)

        if scene_name.lower() == "swarm":
            o = [condition["Kappa"]] * len(dX)
            d = [condition["Density"]] * len(dX)
            mu = [condition["Mu"]] * len(dX)
            spe = [condition["LocustSpeed"]] * len(dX)
            du = [condition["duration"]] * len(dX)
        elif scene_name.lower() == "choice":
            if conditions[id][0]["agent"] == "LeaderLocust":
                o = ["gn_locust"] * len(dX)
            elif conditions[id][0]["agent"] == "":
                o = ["empty_trial"] * len(dX)
            d = [conditions[id][0]["distance"]] * len(dX)
            du = [conditions[id][1]] * len(dX)
            f_angle = [conditions[id][0]["heading_angle"]] * len(dX)
            mu = [conditions[id][0]["walking_direction"]] * len(dX)
            spe = [conditions[id][0]["agent_speed"]] * len(dX)

        groups = [growth_condition] * len(dX)
        df_curated = pd.DataFrame(
            {"X": dX, "Y": dY, "fname": f, "mu": mu, "agent_speed": spe, "duration": du}
        )
        if "elapsed_time" in locals():
            df_curated["ts"] = list(elapsed_time)
        if "temperature" in locals():
            df_curated["temperature"] = list(temperature)
            df_curated["humidity"] = list(humidity)
        if scene_name.lower() == "swarm":
            df_curated["density"] = d
            df_curated["kappa"] = o
        elif scene_name.lower() == "choice":
            df_curated["object_type"] = o
            df_curated["initial_distance"] = d
            df_curated["heading_angle"] = f_angle
            f_angle = [f_angle[0]]

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

            elif scene_name.lower() == "swarm":
                Warning("work in progress")
            if "agent_dX" in locals():
                df_agent = pd.DataFrame(
                    {
                        "X": np.float32(agent_dX),
                        "Y": np.float32(agent_dY),
                        "fname": [fchop] * len(agent_dX),
                        "mu": mu,
                        "agent_speed": spe,
                    }
                )
                if scene_name.lower() == "swarm":
                    print(
                        "there is a unsovled bug about how to name the number of agent"
                    )
                    df_agent["agent_no"] = d
                elif scene_name.lower() == "choice":
                    df_agent["agent_no"] = [0] * len(
                        agent_dX
                    )  # need to figure out a way to solve multiple agents situation. The same method should be applied in the Swarm scene

        df_summary = pd.DataFrame(
            {
                "fname": [f[0]],
                "loss": [loss],
                "mu": [mu[0]],
                "agent_speed": [spe[0]],
                "groups": [groups[0]],
                "mean_angle": [np.float32(meanAngle)],
                "vector": [np.float32(meanVector)],
                "variance": [np.float32(std)],
                "distX": [dX[-1]],
                "distTotal": [tdist],
                "sin": [np.float32(VecSin)],
                "cos": [np.float32(VecCos)],
                "duration": [du[0]],
            }
        )
        if scene_name.lower() == "swarm":
            df_summary["density"] = d
            df_summary["kappa"] = o
        elif scene_name.lower() == "choice":
            df_summary["object_type"] = o
            df_summary["initial_distance"] = d
            df_summary["heading_angle"] = f_angle

        if plotting_trajectory == True:
            if scene_name.lower() == "swarm":
                if df_summary["density"][0] > 0:
                    ## if using plot instead of scatter plot
                    # ax2.plot(
                    #     dX, dY, color=np.arange(len(dY)), alpha=df_curated.iloc[id]["alpha"]
                    # )
                    ##blue is earlier colour and yellow is later colour
                    ax2.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                        alpha=df_summary["kappa"].map(alpha_dictionary)[0],
                    )
                else:
                    ## if using plot instead of scatter plot
                    # ax1.plot(
                    #     dX, dY, alpha=df_curated.iloc[id]["alpha"]
                    # )
                    ax1.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                        alpha=df_summary["kappa"].map(alpha_dictionary)[0],
                    )
            elif scene_name.lower() == "choice":
                if df_summary["object_type"][0] == "empty_trial":
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
        # plt.show()
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
    ## here to load simulated agent's data
    for i in range(num_vr):
        scene_name = analysis_methods.get("experiment_name")
        if scene_name.lower() == "swarm":
            agent_pattern = f"*SimulatedLocustsVR{i+1}*"
        elif scene_name.lower() == "choice":
            agent_pattern = "*Leader*"
        found_result = find_file(thisDir, agent_pattern)
        if found_result is None:
            return print(f"file with {agent_pattern} not found")
        # elif scene_name.lower() == "choice" and i > 0:
        elif scene_name.lower() == "choice" and "ts_simulated_animal" in locals():
            print(
                "Information about simulated locusts are shared across rigs in the choice scene, so start analysing focal animals"
            )
        else:
            ts_simulated_animal = []
            x_simulated_animal = []
            y_simulated_animal = []
            conditions = []
            if isinstance(found_result, list):
                print(
                    f"Analyze {agent_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )
                json_pattern = "*sequenceConfig.json"
                json_file = find_file(thisDir, json_pattern)
                with open(json_file, "r") as f:
                    print(f"load conditions from file {json_file}")
                    tmp = json.loads(f.read())
                for idx in range(len(tmp["sequences"])):
                    this_file = found_result[idx]
                    # for idx, this_file in enumerate(found_result):
                    ts, x, y, condition = read_simulated_data(
                        this_file, analysis_methods
                    )
                    ts_simulated_animal.append(ts)
                    x_simulated_animal.append(x)
                    y_simulated_animal.append(y)
                    if scene_name.lower() == "swarm":
                        condition["duration"] = tmp["sequences"][idx]["duration"]
                    conditions.append(condition)

            elif len(found_result.stem) > 0:
                ts, x, y, condition = read_simulated_data(
                    found_result, analysis_methods
                )
                if scene_name.lower() == "choice":
                    ts_simulated_animal = ts
                    x_simulated_animal = x
                    y_simulated_animal = y
                    conditions = condition
                else:
                    ts, x, y, condition = read_simulated_data(
                        found_result, analysis_methods
                    )
                    ts_simulated_animal.append(ts)
                    x_simulated_animal.append(x)
                    y_simulated_animal.append(y)
                    conditions.append(condition)

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
                        heading_direction_focal_animal,
                        x_focal_animal,
                        y_focal_animal,
                        ts_focal_animal,
                    ) = analyse_focal_animal(
                        this_file,
                        analysis_methods,
                        ts_simulated_animal,
                        x_simulated_animal,
                        y_simulated_animal,
                        conditions,
                        tem_df,
                    )
            elif len(found_result.stem) > 0:
                (
                    heading_direction_focal_animal,
                    x_focal_animal,
                    y_focal_animal,
                    ts_focal_animal,
                ) = analyse_focal_animal(
                    found_result,
                    analysis_methods,
                    ts_simulated_animal,
                    x_simulated_animal,
                    y_simulated_animal,
                    conditions,
                    tem_df,
                )


if __name__ == "__main__":
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240818_170807"
    thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240824_143943"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240826_150826"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_171158"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_151537"
    # thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\archive\20240905_193855"
    # thisDir = r"D:\MatrexVR_grass1_Data\RunData\20240907_142802"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
