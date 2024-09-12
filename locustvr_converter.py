# this is a file to convert data from matrexVR to locustVR.
# Input: csv file, gz csv file from matrexVR
# output: h5 file that stores single animal's response in multiple conditions
import time
import pandas as pd
import numpy as np
import os, gzip, re, csv, json, sys
from pathlib import Path
from threading import Lock
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file, find_nearest
from funcs import *

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
        print("work in progress")
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
        simulated_speed = df.columns[10]
        density = int(n_locusts.split(":")[1]) / (
            int(boundary_size.split(":")[1]) ** 2 / 10000
        )
        conditions = {
            "Density": density,
            mu.split(":")[0]: int(mu.split(":")[1]),
            kappa.split(":")[0]: float(kappa.split(":")[1]),
            simulated_speed.split(":")[0]: float(simulated_speed.split(":")[1]),
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
        vr_pattern = "*_Choice_*.json"
        found_result = find_file(thisDir, vr_pattern)
        if found_result is None:
            return print(f"file with {vr_pattern} not found")
        else:
            condition_dict = {}
            if isinstance(found_result, list):
                print(
                    f"Analyze {vr_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
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
                            "simulated_speed": tmp["objects"][0]["speed"],
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
                        "simulated_speed": tmp["objects"][0]["speed"],
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
            if (
                i == 0
            ):  ## need to add this condition because I hardcode to make the first empty scene 240 sec
                meta_condition = (this_condition, 240)
            else:
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


def analyse_focal_animal(
    this_file,
    analysis_methods,
    ts_simulated_animal,
    x_simulated_animal,
    y_simulated_animal,
    conditions,
):
    # track_ball_radius = analysis_methods.get("trackball_radius_cm")
    # monitor_fps = analysis_methods.get("monitor_fps")
    # camera_fps = analysis_methods.get("camera_fps")
    scene_name = analysis_methods.get("experiment_name")
    alpha_dictionary = {0.1: 0.2, 1.0: 0.4, 10.0: 0.6, 100000.0: 1}
    analyze_one_session_only = True
    BODY_LENGTH = analysis_methods.get("body_length")
    growth_condition = analysis_methods.get("growth_condition")
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset")
    time_series_analysis = analysis_methods.get("time_series_analysis")
    heading_direction_across_trials = []
    x_across_trials = []
    y_across_trials = []
    ts_across_trials = []
    if type(this_file) == str:
        this_file = Path(this_file)
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    # replace 0.0 with np.nan since there are probably generated when switching to the scene
    df["GameObjectPosX"].replace(0.0, np.nan, inplace=True)
    df["GameObjectPosZ"].replace(0.0, np.nan, inplace=True)
    df["GameObjectRotY"].replace(0.0, np.nan, inplace=True)
    df["Current Time"] = pd.to_datetime(
        df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    experiment_id = df["VR"][0] + " " + str(df["Current Time"][0]).split(".")[0]
    experiment_id = re.sub(r"\s+", "_", experiment_id)
    experiment_id = re.sub(r":", "", experiment_id)
    curated_file_path = this_file.parent / f"{experiment_id}_XY.h5"
    summary_file_path = this_file.parent / f"{experiment_id}_score.h5"

    if time_series_analysis:
        print(
            "will delete here later as both analysis needs to deal with overwriting curated dataset"
        )
    else:
        if overwrite_curated_dataset == True and curated_file_path.is_file():
            curated_file_path.unlink()
            try:
                summary_file_path.unlink()
            except OSError as e:
                # If it fails, inform the user.
                print("Error: %s - %s." % (e.filename, e.strerror))
    if analysis_methods.get("plotting_trajectory") == True:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(18, 7), tight_layout=True
        )
        ax1.set_title("ISI")
        ax2.set_title("Trial")
    for id in range(len(conditions)):
        this_range = (df["CurrentStep"] == id) & (df["CurrentTrial"] == 0)
        this_current_time = df["Current Time"][this_range]
        if len(this_current_time) == 0:
            break
        fchop = str(this_current_time.iloc[0]).split(".")[0]
        fchop = re.sub(r"\s+", "_", fchop)
        fchop = re.sub(r":", "", fchop)
        # heading_direction = df["GameObjectRotY"][this_range]
        x = df["GameObjectPosX"][this_range]
        y = df["GameObjectPosZ"][this_range]
        xy = np.vstack((x.to_numpy(), y.to_numpy()))
        # since I introduced nan earlier for the switch scene, I need to fill them with some values otherwise, smoothing methods will fail
        xy = bfill(xy)
        ts = df["Current Time"][this_range]
        trial_no = df["CurrentTrial"][this_range]
        if len(trial_no.value_counts()) > 1 & analyze_one_session_only == True:
            break
        if time_series_analysis:
            print("work in progress")
            elapsed_time = (ts - ts.min()).dt.total_seconds()
            if analysis_methods.get("filtering_method") == "sg_filter":
                X = savgol_filter(xy[0], 59, 3, axis=0)
                Y = savgol_filter(xy[1], 59, 3, axis=0)
            else:
                X = xy[0]
                Y = xy[1]
            travel_distance_fbf = np.sqrt(
                np.add(np.square(np.diff(X)), np.square(np.diff(Y)))
            )  ##need to discuss with Pavan whether it is fair to use Unity clock as elapsed time to calculate speed
            loss = np.nan
        else:
            ##need to think about whether applying removeNoiseVR only to spatial discretisation or general
            loss, X, Y = removeNoiseVR(xy[0], xy[1])
            loss = 1 - loss
            if len(X) == 0:
                print("all is noise")
                continue

        rX, rY = rotate_vector(X, Y, -90 * np.pi / 180)
        if time_series_analysis:
            (dX, dY) = (np.array(rX), np.array(rY))
        else:
            newindex = diskretize(rX, rY, BODY_LENGTH)
            dX = np.array([rX[i] for i in newindex]).T
            dY = np.array([rY[i] for i in newindex]).T
        angles = np.array(ListAngles(dX, dY))

        c = np.cos(angles)
        s = np.sin(angles)
        if len(angles) == 0:
            (xm, ym, meanAngle, meanVector, sin, cos) = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
        else:
            xm = np.sum(c) / len(angles)
            ym = np.sum(s) / len(angles)

            meanAngle = atan2(ym, xm)
            meanVector = np.sqrt(np.square(np.sum(c)) + np.square(np.sum(s))) / len(
                angles
            )
            sin = meanVector * np.sin(meanAngle)
            cos = meanVector * np.cos(meanAngle)

        std = np.sqrt(2 * (1 - meanVector))
        if time_series_analysis:
            tdist = np.sum(
                travel_distance_fbf
            )  ##note this distance can be a lot larger than calculating with spatial discretisation
        else:
            tdist = len(dX) * BODY_LENGTH

        f = [fchop] * len(dX)
        loss = [loss] * len(dX)
        if scene_name.lower() == "swarm":
            o = [conditions[id]["Kappa"]] * len(dX)
            d = [conditions[id]["Density"]] * len(dX)
            mu = [conditions[id]["Mu"]] * len(dX)
            spe = [conditions[id]["LocustSpeed"]] * len(dX)
        elif scene_name.lower() == "choice":
            if conditions[id][0]["agent"] == "LeaderLocust":
                o = ["gn_locust"] * len(dX)
            elif conditions[id][0]["agent"] == "":
                o = ["empty_trial"] * len(dX)
            d = [conditions[id][0]["distance"]] * len(dX)
            du = [conditions[id][1]] * len(dX)
            f_angle = [conditions[id][0]["heading_angle"]] * len(dX)
            mu = [conditions[id][0]["walking_direction"]] * len(dX)
            spe = [conditions[id][0]["simulated_speed"]] * len(dX)

        groups = [growth_condition] * len(dX)
        if time_series_analysis:
            print("since the length is different, I should also reorganise the timing")
        else:
            df_curated = pd.DataFrame(
                {
                    "X": dX,
                    "Y": dY,
                    "fname": f,
                    "loss": loss,
                    "density": d,
                    "mu": mu,
                    "agent_speed": spe,
                    "groups": groups,
                }
            )
            if scene_name.lower() == "swarm":
                df_curated["density"] = d
                df_curated["order"] = o
            elif scene_name.lower() == "choice":
                df_curated["object_type"] = o
                df_curated["initial_distance"] = d
                df_curated["heading_angle"] = f_angle
                f_angle = [f_angle[0]]
                df_curated["duration"] = du

        f = [f[0]]
        loss = [loss[0]]
        o = [o[0]]
        d = [d[0]]
        mu = [mu[0]]
        spe = [spe[0]]
        groups = [groups[0]]
        V = [meanVector]
        MA = [meanAngle]
        ST = [std]
        lX = [dX[-1]]
        tD = [tdist]
        sins = [sin]
        coss = [cos]
        du = [du[0]]

        df_summary = pd.DataFrame(
            {
                "fname": f,
                "loss": loss,
                "mu": mu,
                "agent_speed": spe,
                "groups": groups,
                "mean_angle": MA,
                "vector": V,
                "variance": ST,
                "distX": lX,
                "distTotal": tD,
                "sin": sins,
                "cos": coss,
            }
        )
        if scene_name.lower() == "swarm":
            df_summary["density"] = d
            df_summary["order"] = o
        elif scene_name.lower() == "choice":
            df_summary["object_type"] = o
            df_summary["initial_distance"] = d
            df_summary["heading_angle"] = f_angle
            df_summary["duration"] = du
        if analysis_methods.get("plotting_trajectory") == True:
            if scene_name.lower() == "swarm":
                if df_summary["density"][0] > 0:
                    # ax2.plot(
                    #     dX, dY, color=np.arange(len(dY)), alpha=df_curated.iloc[id]["alpha"]
                    # )
                    ##blue is earlier colour and yellow is later colour
                    ax2.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                        alpha=df_summary["order"].map(alpha_dictionary)[0],
                    )
                else:
                    # ax1.plot(
                    #     dX, dY, alpha=df_curated.iloc[id]["alpha"]
                    # )
                    ax1.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                        alpha=df_summary["order"].map(alpha_dictionary)[0],
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

        #######################Sections to save data
        if analysis_methods.get("debug_mode") == False:
            with lock:
                if time_series_analysis:
                    print(
                        "if using time series analysis, there should be a better way to organise the data"
                    )
                else:
                    store = pd.HDFStore(curated_file_path)
                    store.append(
                        "name_of_frame",
                        df_curated,
                        format="t",
                        data_columns=df_curated.columns,
                    )
                    store.close()
                store = pd.HDFStore(summary_file_path)
                store.append(
                    "name_of_frame",
                    df_summary,
                    format="t",
                    data_columns=df_summary.columns,
                )
                store.close()

        heading_direction_across_trials.append(angles)
        x_across_trials.append(x)
        y_across_trials.append(y)
        if time_series_analysis:
            ts_across_trials.append(elapsed_time)
        else:
            ts_across_trials.append(ts)
    trajectory_fig_path = this_file.parent / f"{experiment_id}_trajectory.png"
    if analysis_methods.get("plotting_trajectory") == True:
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
    num_vr = 4
    for i in range(num_vr):
        # i = i + 1
        scene_name = analysis_methods.get("experiment_name")
        if scene_name.lower() == "swarm":
            vr_pattern = f"*SimulatedLocustsVR{i+1}*"
        elif scene_name.lower() == "choice":
            vr_pattern = f"*Leader*"
        found_result = find_file(thisDir, vr_pattern)
        if found_result is None:
            return print(f"file with {vr_pattern} not found")
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
                    f"Analyze {vr_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )

                for this_file in found_result:
                    ts, x, y, condition = read_simulated_data(
                        this_file, analysis_methods
                    )
                    ts_simulated_animal.append(ts)
                    x_simulated_animal.append(x)
                    y_simulated_animal.append(y)
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
        locust_pattern = f"*_VR{i+1}*"
        found_result = find_file(thisDir, locust_pattern)
        if found_result is None:
            return print(f"file with {locust_pattern} not found")
        else:
            if isinstance(found_result, list):
                print(
                    f"Analyze {locust_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
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
                )


if __name__ == "__main__":
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240818_170807"
    # thisDir = r"D:\MatrexVR_Swarm_Data\RunData\20240826_150826"
    thisDir = r"D:\MatrexVR_blackbackground_Data\RunData\20240904_151537"
    json_file = r"C:\Users\neuroPC\Documents\GitHub\UnityDataAnalysis\analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
