# this is a file to convert data from matrexVR to locustVR so that we can use Sercan's code to analyse locust's alignment score.
import time
import pandas as pd
import numpy as np
import os, gzip, re, csv, json, sys
from pathlib import Path

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file, find_nearest


def read_simulated_data(this_file, analysis_methods):
    print("read simulated data")
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    print(df.columns)
    n_locusts = df.columns[6]
    boundary_size = df.columns[7]
    mu = df.columns[8]
    kappa = df.columns[9]
    simulated_speed = df.columns[10]
    conditions = {
        n_locusts.split(":")[0]: int(n_locusts.split(":")[1]),
        boundary_size.split(":")[0]: int(boundary_size.split(":")[1]),
        mu.split(":")[0]: int(mu.split(":")[1]),
        kappa.split(":")[0]: int(kappa.split(":")[1]),
        simulated_speed.split(":")[0]: int(simulated_speed.split(":")[1]),
    }
    if len(df) > 0:
        ts = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
        x = df["X"]
        y = df["Z"]
    else:
        ts = pd.to_datetime(this_file.stem[0:19], format="%Y-%m-%d_%H-%M-%S")
        x = None
        y = None
    return ts, x, y, conditions


def align_matrex_data(this_file, analysis_methods, ts_all, x_all, y_all):
    print("read locust data")
    track_ball_radius = analysis_methods.get("trackball_radius")
    monitor_fps = analysis_methods.get("monitor_fps")
    camera_fps = analysis_methods.get("camera_fps")
    if this_file.suffix == ".gz":
        with gzip.open(this_file, "rb") as f:
            df = pd.read_csv(f)
    elif this_file.suffix == ".csv":
        with open(this_file, mode="r") as f:
            df = pd.read_csv(f)
    print(df.columns)
    df["Current Time"] = pd.to_datetime(
        df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    df.set_index(df["Current Time"], inplace=True)
    for id in range(len(ts_all)):
        this_ts = ts_all[id]
        try:
            df[df.index == this_ts]
            try:
                next_ts = ts_all[id + 1]
                this_range = (df.index > this_ts) & (df.index < next_ts[0])
            except:
                print(df.iloc[len(df) - 1, :])
                this_range = (df.index > this_ts) & (df.index < df.index[len(df) - 1])
        except:
            this_range = (df.index > this_ts[0]) & (
                df.index < this_ts[len(this_ts) - 1]
            )
        df["SensRotY"][this_range]
        df["SensPosX"][this_range]
        df["SensPosY"][this_range]
    # ts = pd.to_datetime(df["Current Time"], format="%Y-%m-%d %H_%M_%S")
    # df["step_distance"] = np.sqrt(
    #     (df["SensPosX"].diff()) ** 2 + (df["SensPosY"].diff()) ** 2
    # )
    # heading_direction = df["SensRotY"]
    # df["step_distance_mm"] = df["step_distance"] * track_ball_radius
    # # calculate time between each step with Current Time: Timestamp('2024-05-16 14:16:35.300000')
    # df["time_diff"] = df["Current Time"].diff()
    # df["time_diff_ms"] = df["time_diff"].dt.total_seconds() * 1000
    # # calculate speed of each step
    # df["speed_mm_s"] = df["step_distance_mm"] / df["time_diff_ms"] * 1000


def preprocess_matrex_data(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    # dat_pattern = "*.dat"
    # found_result = find_file(thisDir, dat_pattern)
    # # raw_data = pd.read_table(found_result, sep="\s+")
    # raw_data = pd.read_csv(found_result, sep=" ", header=0)
    # raw_data.iloc[:, -4:-2]
    num_vr = 4
    for i in range(num_vr):
        vr_pattern = f"*SimulatedLocustsVR{i+1}*"
        found_result = find_file(thisDir, vr_pattern)
        if found_result is None:
            return print(f"file with {vr_pattern} not found")
        else:
            ts_all = []
            x_all = []
            y_all = []
            if isinstance(found_result, list):
                print(
                    f"Analyze {vr_pattern} data which come with multiple trials of vr models. Use a for-loop to go through them"
                )

                for this_file in found_result:
                    ts, x, y, conditions = read_simulated_data(
                        this_file, analysis_methods
                    )
                    ts_all.append(ts)
                    x_all.append(x)
                    y_all.append(y)

            elif len(found_result.stem) > 0:
                ts, x, y, conditions = read_simulated_data(
                    found_result, analysis_methods
                )
                ts_all.append(ts)
                x_all.append(x)
                y_all.append(y)

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
                    align_matrex_data(this_file, analysis_methods, ts_all, x_all, y_all)
            elif len(found_result.stem) > 0:
                align_matrex_data(found_result, analysis_methods, ts_all, x_all, y_all)


if __name__ == "__main__":
    thisDir = r"C:\Users\neuroPC\Documents\20240818_134521"
    json_file = r".\analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
