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

import time,os
import pandas as pd
import numpy as np
import gzip, re, json, sys
from pathlib import Path
from threading import Lock
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter
import h5py
from deepdiff import DeepDiff
from pprint import pprint
import pyarrow.parquet as pq
import pyarrow as pa
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
if os.name == 'nt':
    sys.path.insert(0, str(parent_dir) + "\\utilities")
else:
    sys.path.insert(0, str(parent_dir) + "/utilities")
from useful_tools import find_file
from data_cleaning import (
    load_temperature_data,
    remove_unreliable_tracking,
    bfill,
    diskretize,
    findLongestConseqSubseq,interp_fill,euclidean_distance
)
from sorting_time_series_analysis import calculate_speed,diff_angular_degree

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
def get_list_depth(lst):
    if isinstance(lst, list):
        return 1 + max(get_list_depth(item) for item in lst)
    else:
        return 0

def save_curated_dataset(
    thisH5,
    timestamp,
    dataset,
    *args,
):

    if type(dataset) == pd.DataFrame:
        ##need to think about whether it makes sense to summarise 4 vrs into one hdf file with different keys, or save them into different files.
        dataset.to_hdf(
            thisH5, key="summary", mode="w"
        )  # better used in dataset that has certain structure
    else:
        with h5py.File(thisH5, "w") as h5file:
            try:
                if isinstance(dataset, list):
                    # num_trials=len(dataset)
                    for id, this_trial_coordinates in enumerate(dataset):
                        trials_id = f"trial{id+1}"
                        if len(timestamp) == len(dataset):
                            timestamp_id = id
                        else:
                            timestamp_id = id * 2
                        tracklets = h5file.create_group(trials_id)
                        tracklets.create_dataset(
                            name="TimeStamp", data=timestamp[timestamp_id]
                        )
                        if len(args) > 0:
                            for this_argument in range(len(args)):
                                tracklets.create_dataset(
                                    name="Heading", data=args[this_argument][id]
                                )
                        else:
                            print("no additional variable to saved")
                        if type(this_trial_coordinates) == list:
                            pass
                        else:
                            this_trial_coordinates = [this_trial_coordinates]
                        for i in range(len(this_trial_coordinates)):
                            tracklet_id = f"XY{i+1}"
                            theses_coordinates = this_trial_coordinates[i]
                            if theses_coordinates.ndim == 2:
                                tracklets.create_dataset(
                                    name=tracklet_id, data=theses_coordinates
                                )
                            elif theses_coordinates.ndim == 3:
                                print(
                                    "presumably the 3rd dimension will be visibility. Not sure how to save it though"
                                )
                            else:
                                print(
                                    "unknown dimension is detected. Do not save this data into the H5 file"
                                )
                elif type(dataset) == pd.DataFrame:
                    print(
                        "maybe dont need this. Save hdf file with pandas dataframe seems to be more easy to understand and efficient"
                    )
                    tracklets = h5file.create_group("summary")
                    tracklets.create_dataset(name="summary", data=dataset)
                    # dataset.to_hdf(h5file, key='summary', mode='w')#better used in dataset that has certain structure
                else:
                    tracklets = h5file.create_group("trial1")
                    tracklets.create_dataset(name="TimeStamp", data=timestamp)
                    theses_coordinates = dataset
                    if theses_coordinates.ndim == 2:
                        tracklets.create_dataset(name="XY1", data=theses_coordinates)
                    elif theses_coordinates.ndim == 3:

                        print(
                            "presumably the 3rd dimension will be visibility. Not sure how to save it though"
                        )
                    else:
                        print(
                            "unknown dimension is detected. Do not save this data into the H5 file"
                        )
                ###should be able to compress the data but let me just store the data first
                # t1.create_dataset(name='Coordinates_XY',data=dataset,chunks=True,compression='gzip',scaleoffset=True,shuffle=True)
            except EOFError:
                h5file.close()


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

def fill_missing_data(df):
    ## this is used to fill in missing points in Unity dataset during scene changes. 
    ## By taking fictrac data as a reference, we can estimate what values unity should have during scene changes
    ## Note, there is 1 or 2 frames delay when logging fictrac and unity data 
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
        else:
            continue

    return df


def reshape_multiagent_data(df, this_object=None):
    if "Timestamp" in df.columns:
        number_of_duplicates = df["Timestamp"].drop_duplicates().shape[0]
        number_of_instances = int(df.shape[0] / number_of_duplicates)
        if number_of_instances == 1 and df.shape[0] > number_of_duplicates:
            df = df.drop_duplicates(
                ignore_index=True
            )  # need to add this line to avoid duplicate entries simply comes from logging the same entry twice. This however, shall also happen when rendering multiple agents. Not sure why I did not see this error in the past
        else:
            pass
        agent_id = np.tile(
            np.arange(number_of_instances) + number_of_instances * this_object,
            number_of_duplicates,
        )
        c_name_list = ["agent" + str(num) for num in agent_id]
        test = pd.concat([df, pd.DataFrame(c_name_list)], axis=1)
        df_values = ["X", "Z", "VisibilityPhase"]
        new_df = test.pivot(index="Timestamp", columns=0, values=df_values)
    elif "Current Time" in df.columns:
        number_of_instances = 2
        trial_length = int(df["Current Time"].shape[0] / number_of_instances)
        agent_id = np.repeat(np.arange(number_of_instances), trial_length)
        c_name_list = ["agent" + str(num) for num in agent_id]
        df.reset_index(inplace=True)
        df = df.drop(["index"], axis=1)
        test = pd.concat([df, pd.DataFrame(c_name_list)], axis=1)
        df_values = [df.columns[1], df.columns[2]]
        new_df = pd.pivot_table(
            test, values=[df.columns[1], df.columns[2]], index="Current Time", columns=0
        )
    else:
        Warning("no columns found")
    return new_df


def prepare_data(df,x_title,y_title,rot_title,this_range=None):
    #this is for collecting data from the fictrac whole dataset
    if this_range is None:
        this_range=df["CurrentTrial"]==0
    ts = df["Current Time"][this_range]
    if ts.empty:
        return None
    x = df[x_title][this_range]
    y = df[y_title][this_range]
    rot_y = df[rot_title][this_range]
    step_id = df["CurrentStep"][this_range]
    session_id= df["CurrentTrial"][this_range]
    xy = np.vstack((x.to_numpy(), y.to_numpy()))
    return ts,xy,rot_y,step_id,session_id


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

    # print(df.columns)
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
                mu.split(":")[0].lower(): float(mu.split(":")[1]),
                kappa.split(":")[0].lower(): float(kappa.split(":")[1]),
                "speed": float(agent_speed.split(":")[1]),
            }
        else:
            mu = these_parameters["mu"]
            kappa = these_parameters["kappa"]
            agent_speed = these_parameters["locustSpeed"]
            conditions = {
                "density": float(density),
                "mu": float(mu),
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

    elif scene_name.lower() == "band" or scene_name.lower() == "kannadi":
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
            conditions["spawnLengthX"] * conditions["spawnLengthZ"] / 10000
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
            # not sure why the datatime is not saved in the result during ISI
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
                    pd.to_datetime(
                        df["Current Time"][df["VR"].str.startswith(conditions["type"])],
                        format="%Y-%m-%d %H:%M:%S.%f",
                    ),
                    df["GameObjectPosX"][df["VR"].str.startswith(conditions["type"])],
                    df["GameObjectPosZ"][df["VR"].str.startswith(conditions["type"])],
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
    temperature_df=None,
):
    monitor_fps = analysis_methods.get("monitor_fps")
    plotting_trajectory = analysis_methods.get("plotting_trajectory", False)
    save_output = analysis_methods.get("save_output", False)
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset", False)
    export_fictrac_data_only = analysis_methods.get("export_fictrac_data_only", False)
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
    elif type(this_file) == list:
        df1 = load_file(this_file[0])
        df2 = load_file(this_file[1])
        this_file=this_file[0]
        df = pd.concat([df1, df2], ignore_index=True)   
    else:
        df = load_file(this_file)
    df["Current Time"] = pd.to_datetime(
        df["Current Time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    df=df.sort_values(by='Current Time',ascending=True)
    df = fill_missing_data(df)
    test = np.where(df["GameObjectRotY"].values == 0)[0]
    longest_unity_gap = findLongestConseqSubseq(test, test.shape[0])
    print(f"longest unfilled gap is {longest_unity_gap}")
    test = np.where(df["SensRotY"].values == 0)[0]
    longest_fictrac_gap = findLongestConseqSubseq(test, test.shape[0])
    print(f"longest unfilled gap is {longest_fictrac_gap}")

    # replace 0.0 with np.nan since they are generated during scene-switching
    ##if upgrading to pandas 3.0 in the future, try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead
    df.replace(
        {
            "GameObjectPosX": {0.0: np.nan},
            "GameObjectPosZ": {0.0: np.nan},
            "GameObjectRotY": {0.0: np.nan},
            "SensPosX": {0.0: np.nan},
            "SensPosY": {0.0: np.nan},
            "SensRotY": {0.0: np.nan},
        },
        inplace=True,
    )

    experiment_id = df["VR"][0] + " " + str(df["Current Time"][0]).split(".")[0]
    experiment_id = re.sub(r"\s+", "_", experiment_id)
    experiment_id = re.sub(r":", "", experiment_id)
    file_suffix = "_full" if time_series_analysis else ""
    curated_file_path = this_file.parent / f"{experiment_id}_XY{file_suffix}.h5"
    summary_file_path = this_file.parent / f"{experiment_id}_score{file_suffix}.h5"
    agent_file_path = this_file.parent / f"{experiment_id}_agent{file_suffix}.h5"
    pa_file_path = this_file.parent / f"{experiment_id}_motion{file_suffix}.parquet"
    # need to think about whether to name them the same regardless analysis methods

    ts,xy,rot_y,step_id,session_id =prepare_data(df,"SensPosX","SensPosY","SensRotY")
    if len(ts) == 0:
        print("empty file")
        return None,None,None,None
    xy=xy*analysis_methods.get("trackball_radius_cm")
    remains, X, Y,mask = remove_unreliable_tracking(xy[0], xy[1], analysis_methods)
    loss = 1 - remains
    elapsed_time = (ts - ts.min()).dt.total_seconds().values
    if time_series_analysis and analysis_methods.get("filtering_method") == "sg_filter":
        X = bfill(X)
        Y = bfill(Y)
        X = savgol_filter(X, 59, 3, axis=0)
        Y = savgol_filter(Y, 59, 3, axis=0)
    else:
        elapsed_time=elapsed_time[:-1][mask]
        step_id=step_id.values[:-1][mask]
        rot_y=rot_y.values[:-1][mask]
    if overwrite_curated_dataset ==True or pa_file_path.is_file()==False:
        pq.write_table(pa.table({"X": X,"Y":Y,"heading_angle":rot_y,"elapsed_time":elapsed_time,"step_id": step_id}), pa_file_path)
    print(f"export {pa_file_path}")
    if export_fictrac_data_only:
        return None,None,None,None
    if temperature_df is None:
        df["Temperature ˚C (ºC)"] = np.nan
        df["Relative Humidity (%)"] = np.nan
    else:
        frequency_milisecond = int(1000 / monitor_fps)
        temperature_df = temperature_df.resample(
            f"{frequency_milisecond}L"
        ).interpolate()  # FutureWarning: 'L' is deprecated and will be removed in a future version, please use 'ms' instead.
        df.set_index("Current Time", drop=False, inplace=True)
        df = df.join(temperature_df.reindex(df.index, method="nearest"))
        del temperature_df

    if overwrite_curated_dataset and summary_file_path.is_file():
        summary_file_path.unlink(missing_ok=True)
        curated_file_path.unlink(missing_ok=True)
        agent_file_path.unlink(missing_ok=True)

    if plotting_trajectory:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), tight_layout=True)
        ax1.set_title("ISI")
        ax2.set_title("Trial")
    (
        summary_across_trials,
        heading_direction_across_trials,
        xy_across_trials,
        agent_across_trials,
        ts_across_trials,
    ) = ([], [], [], [], [])
    for id, condition in enumerate(conditions):
        this_range = (df["CurrentStep"] == id) & (df["CurrentTrial"] == 0)
        ts,xy,rot_y,_,session_id = prepare_data(df,"GameObjectPosX","GameObjectPosZ","GameObjectRotY",this_range)
        if len(ts) == 0:
            break
        elif len(session_id.value_counts()) > 1 & analyze_one_session_only == True:
            break
        fchop = ts.iloc[0].strftime("%Y-%m-%d_%H%M%S")
        ## clean up bad tracking of fictrac
        remains, X, Y,mask = remove_unreliable_tracking(xy[0], xy[1], analysis_methods)
        if len(X) == 0:
            print("all is noise")
            continue
        loss = 1 - remains
        elapsed_time = (ts - ts.min()).dt.total_seconds().values
        if time_series_analysis and analysis_methods.get("filtering_method") == "sg_filter":
            X = bfill(X)
            Y = bfill(Y)
            X = savgol_filter(X, 59, 3, axis=0)
            Y = savgol_filter(Y, 59, 3, axis=0)
            angles_rad = np.radians(-rot_y.values)
        else:
            if remains<1.0:
                print("hi, an unreliable tracking was found")
            ts=ts[:-1][mask]
            elapsed_time=elapsed_time[:-1][mask]
            rot_y=rot_y.values[:-1][mask]
            angles_rad = np.radians(
                -rot_y)  # turn negative to acount for Unity's axis and turn radian

        ## align focal animal's timestamp with agents' timestamp
        list_depth=get_list_depth(df_simulated_animal)
        if scene_name == "choice" and id % 2 > 0:
            if type(df_simulated_animal[id]) == list:
                df_simulated = df_simulated_animal[id][0]
            else:
                df_simulated = df_simulated_animal[id]
                if "Current Time" in df_simulated.columns:
                    df_simulated.set_index("Current Time", inplace=True)            
            if ts.index.inferred_type == "datetime64":
                if df_simulated.index.is_monotonic_increasing ==False or ts.index.is_monotonic_increasing ==False:
                    df_simulated = df_simulated.sort_index()## somehow the index is not sorted
                    ts = ts.sort_index()
                df_simulated = df_simulated.reindex(ts.index, method="nearest")
            else:
                df_simulated = df_simulated.reindex(ts, method="nearest") ## when the temperature data is not available, ts index would be normal
        elif (
            scene_name != "choice"
            and isinstance(df_simulated_animal[id], pd.DataFrame) == True
        ):
            these_simulated_agents = df_simulated_animal[id]
            these_simulated_agents = these_simulated_agents.reindex(
                ts.index, method="nearest"
            )#align the timestamp with focal animal
            if scene_name.lower() == "kannadi":
                num_agent_game_object=1#int(df_simulated_animal[id].shape[1]/3)#pre choice phase of kannadi experiment goes here
                # for i in range(num_agent_game_object):
                #     these_simulated_agents          
            else:
                num_agent_game_object=len(df_simulated_animal[id])
        elif (
            scene_name != "choice"
            and isinstance(df_simulated_animal[id], pd.DataFrame) == False
        ):
            if list_depth==1:
                pass
            elif isinstance(df_simulated_animal[id][0], pd.DataFrame) == True:
                num_agent_game_object=len(df_simulated_animal[id])
                reindex_temp_list=[]
                for i in range(num_agent_game_object):
                    reindex_temp_list.append(df_simulated_animal[id][i].reindex(ts.index, method="nearest"))
                #these_simulated_agents = df_simulated_animal[id][0].reindex(ts.index, method="nearest")
                #plus1 = df_simulated_animal[id][1].reindex(ts.index, method="nearest")
                these_simulated_agents=pd.concat(reindex_temp_list,axis=1)
                #these_simulated_agents=these_simulated_agents.join(plus1)

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
            #num_spatial_decision = len(angles)
        else:
            newindex = diskretize(list(rXY[0]), list(rXY[1]), BODY_LENGTH3)
            dX = rXY[0][newindex]
            dY = rXY[1][newindex]
            temperature = df.iloc[newindex]["Temperature ˚C (ºC)"].values
            humidity = df.iloc[newindex]["Relative Humidity (%)"].values
            elapsed_time=elapsed_time[newindex]
            if "these_simulated_agents" in locals() and scene_name.lower() != "choice":
                these_simulated_agents=these_simulated_agents.iloc[newindex,:]
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
            meanVector = (
                np.sqrt(np.square(np.nansum(c)) + np.square(np.nansum(s)))
                / num_spatial_decision
            )
            VecSin = meanVector * np.sin(meanAngle)
            VecCos = meanVector * np.cos(meanAngle)
        std = np.sqrt(2 * (1 - meanVector))

        tdist = (
            np.sum(euclidean_distance(X,Y))
            if time_series_analysis
            else len(dX) * BODY_LENGTH3
        )  ##The distance calculated based on spatial discretisation should be the shortest

        f = [fchop] * len(dX)
        different_key_list = []
        if isinstance(condition, list):
            diff_con = DeepDiff(condition[0], condition[1])
            pprint(diff_con)
            for different_key in diff_con.affected_root_keys:
                different_key_list.append(different_key)
            condition = condition[0]
            if "type" in different_key_list:
                value1 = diff_con["values_changed"]["root['type']"]["old_value"]
                value2 = diff_con["values_changed"]["root['type']"]["new_value"]
                condition["type"] = f"{value1}_x_{value2}"

                
        if 'speed' in condition.keys():
            spe = [float(condition["speed"])] * len(dX)
            mu = [float(condition["mu"])] * len(dX)
        else:
            spe = [np.nan] * len(dX)
            mu = [np.nan] * len(dX)
        du = [condition["duration"]] * len(dX)
        if scene_name.lower() == "swarm" or scene_name.lower() == "band":
            order = [float(condition["kappa"])] * len(dX)
            density = [condition["density"]] * len(dX)
        if scene_name.lower() == "choice" or scene_name.lower() == "band":
            polar_angle = [float(condition["polar_angle"])] * len(dX)
            radial_distance = [float(condition["radial_distance"])] * len(dX)

        if scene_name.lower() == "choice":
            if condition["type"] == "":
                object_type = ["empty_trial"] * len(dX)
            else:
                object_type = [condition["type"]] * len(dX)

        if scene_name.lower() == "band":
            voff = [condition["visibleOffDuration"]] * len(dX)
            von = [condition["visibleOnDuration"]] * len(dX)
            f_angle = [condition["rotationAngle"]] * len(dX)
            object_type = [condition["type"]] * len(dX)
        if scene_name.lower() == "kannadi":
            if "type" in condition.keys():
                object_type = [condition["type"]] * len(dX)
            elif 'kannadiTilePrefab' in condition.keys():
                object_type = [condition['kannadiTilePrefab']] * len(dX)
            else:
                object_type = ["empty_trial"] * len(dX)


        df_curated = pd.DataFrame(
            {
                "X": dX,
                "Y": dY,
                "heading": angles,
                "fname": f,
                "mu": mu,
                "speed": spe,
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
                num_agent = int(df_simulated.shape[1] / 2)
                agent_dXY_list = []
                for i in range(0, num_agent):
                    agent_xy = np.vstack(
                        (
                            df_simulated.iloc[:, i].values,
                            df_simulated.iloc[:, i + num_agent].values,
                        )
                    )
                    agent_rXY = rot_matrix @ np.vstack((agent_xy[0], agent_xy[1]))
                    if time_series_analysis:
                        (agent_dX, agent_dY) = (agent_rXY[0], agent_rXY[1])
                    else:
                        agent_dX = agent_rXY[0][newindex]
                        agent_dY = agent_rXY[1][newindex]
                    agent_dXY_list.append(np.vstack((agent_dX, agent_dY)))

        # elif (
        #     isinstance(df_simulated_animal[id], pd.DataFrame) == True
        #     and "these_simulated_agents" in locals()
        # ):
        elif "these_simulated_agents" in locals():
            pivot_table_column=int(len(these_simulated_agents.columns)/len(these_simulated_agents.columns.get_level_values(0).unique()))
            num_agent_per_object = int(these_simulated_agents.shape[1] / pivot_table_column/num_agent_game_object)
            agent_dXY_list = []
            if isinstance(df_simulated_animal[id], pd.DataFrame) == True or isinstance(df_simulated_animal[id][0], pd.DataFrame) == True:
                for j in range(num_agent_game_object):
                    for i in range(num_agent_per_object):
                        agent_xy = np.vstack(
                            (
                                these_simulated_agents.iloc[:,0*num_agent_per_object+j*num_agent_per_object*pivot_table_column+i].values,
                                these_simulated_agents.iloc[:,1*num_agent_per_object+j*num_agent_per_object*pivot_table_column+i].values,
                            )
                        )
                        agent_rXY = rot_matrix @ np.vstack((agent_xy[0], agent_xy[1]))
                        # if time_series_analysis:
                        #     (agent_dX, agent_dY) = (agent_rXY[0], agent_rXY[1])
                        # else:
                        #     agent_dX = agent_rXY[0][newindex]
                        #     agent_dY = agent_rXY[1][newindex]
                        (agent_dX, agent_dY) = (agent_rXY[0], agent_rXY[1])
                        agent_dXY_list.append(np.vstack((agent_dX, agent_dY)))
            del these_simulated_agents
            
        if "agent_dX" in locals() and scene_name.lower() == "choice":
            df_agent_list = []
            for k in range(len(agent_dXY_list)):
                this_agent_dx = agent_dXY_list[k][0]
                this_agent_dy = agent_dXY_list[k][1]
                df_agent = pd.DataFrame(
                    {
                        "X": this_agent_dx,
                        "Y": this_agent_dy,
                        "fname": [fchop] * len(this_agent_dx),
                        "mu": mu,
                        "speed": spe,
                    }
                )
                # df_agent["agent_type"] = [value_list[k]] * len(
                #     this_agent_dx
                # )
                df_agent_list.append(
                    df_agent
                )  # need to figure out a way to solve multiple agents situation. The same method should be applied in the Swarm scene
        elif "agent_dX" in locals() and scene_name.lower() == "swarm":
            df_agent = pd.DataFrame(
                {
                    "X": agent_dX,
                    "Y": agent_dY,
                    "fname": [fchop] * len(agent_dX),
                    "mu": mu,
                    "speed": spe,
                }
            )
            print("there is a unfixed bug about how to name the number of agent")
            df_agent["agent_num"] = density
        elif "agent_dX" in locals() and scene_name.lower() == "band" or "agent_dX" in locals() and scene_name.lower() == "kannadi":
            df_agent_list = []
            for k in range(len(agent_dXY_list)):
                this_agent_dx = agent_dXY_list[k][0]
                this_agent_dy = agent_dXY_list[k][1]
                #if num_agent_game_object>1 and num_agent_per_object>1 and k>0:
                df_agent = pd.DataFrame(
                    {
                        "X": this_agent_dx,
                        "Y": this_agent_dy,
                        "fname": [fchop] * len(this_agent_dx),
                        # "mu": mu,
                        # "speed": spe,
                    }
                )
                # df_agent["agent_type"] = [value_list[k]] * len(
                #     this_agent_dx
                # )
                df_agent_list.append(
                    df_agent
                )  # need to figure out a way to solve multiple agents situation. The same method should be applied in the Swarm scene
        df_summary = pd.DataFrame(
            {
                "fname": [f[0]],
                "loss": loss,
                "mu": mu[0],
                "speed": spe[0],
                "groups": [growth_condition],
                "mean_angle": meanAngle,
                "vector": meanVector,
                "variance": std,
                "distX": dX[-1],
                "distTotal": tdist,
                "sin": VecSin,
                "cos": VecCos,
                "duration": du[0],
                "rotation_gain":condition['rotation_gain'],
                "translation_gain":condition['translation_gain']
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
        if scene_name.lower()=='kannadi':
            df_summary["type"] = [object_type[0]]
            if "numberOfRings" in condition.keys():
                df_summary["numberOfRings"] = [condition["numberOfRings"]]
                df_summary["watchIndex"] = [condition["watchIndex"]]
            else:
                df_summary["numberOfRings"] = [np.nan]
                df_summary["watchIndex"] = [np.nan]
                df_summary["moveWithTransform"] = [condition["moveWithTransform"]]
            df_summary["boundaryLengthX"] = [condition["boundaryLengthX"]]
            df_summary["boundaryLengthZ"] = [condition["boundaryLengthZ"]]
            df_summary["hexRadius"] = [condition["hexRadius"]]

            # df_summary.drop(['mu', 'speed'], axis=1)
            # df_curated.drop(['mu', 'speed'], axis=1)

        if plotting_trajectory == True:
            if scene_name.lower() == "swarm" or scene_name.lower() == "band":
                if df_summary["density"][0] > 0:
                    ## if using plot instead of scatter plot
                    ax2.plot(dX, dY)
                    ##blue is earlier colour and yellow is later colour
                else:
                    ax1.plot(dX, dY)
            elif scene_name.lower() == "choice" or scene_name.lower() == 'kannadi':
                if df_summary["type"][0] == "empty_trial" or 'kannadiTilePrefab' in condition.keys():
                    ax1.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                    )
                    if scene_name.lower() == 'kannadi':
                        ax1.set_title('Kannadi phase')
                else:
                    ax2.scatter(
                        dX,
                        dY,
                        c=np.arange(len(dY)),
                        marker=".",
                    )
                    if "agent_dX" in locals():
                        for j in range(len(df_agent_list)):
                            this_pd = df_agent_list[j]
                            ax2.plot(
                                this_pd["X"].values,
                                this_pd["Y"].values,
                                c="k",
                                linewidth=1,
                            )
        #######################Sections to save data
        if save_output == True:
            with lock:
                if "df_agent" in locals():
                    file_list = [curated_file_path, agent_file_path]
                    data_frame_list = [df_curated, df_agent_list]
                    hdf_keys = ["focal_animal", ""]
                else:
                    file_list = [curated_file_path]
                    data_frame_list = [df_curated]
                    hdf_keys = ["focal_animal"]
                for this_name, this_pd, this_hdf in zip(
                    file_list, data_frame_list, hdf_keys
                ):
                    store = pd.HDFStore(this_name)
                    if len(hdf_keys) > 1 and this_name == agent_file_path:
                        value_list = [object_type[0]] * len(df_agent_list)
                        if scene_name.lower() != "swarm" and num_agent_game_object>=1 and num_agent_per_object>1:# change the condition here to hopefully include the condition of both kannadi and band where one game object is used  
                            agent_key = value_list[0]
                            this_pd = pd.concat(df_agent_list,axis=0)
                            store.append(
                                agent_key,
                                this_pd,
                                format="t",
                                data_columns=this_pd.columns,
                            )
                        
                        else:
                            for j in range(len(df_agent_list)):
                                agent_key = value_list[j]
                                this_pd = df_agent_list[j]
                                store.append(
                                    agent_key,
                                    this_pd,
                                    format="t",
                                    data_columns=this_pd.columns,
                                )
                        store.close()
                    else:
                        if len(different_key_list) > 0:
                            for i in range(len(different_key_list)):
                                this_different_key = different_key_list[i]
                                if (
                                    this_different_key in this_pd.columns
                                    and this_different_key != "type"
                                ):
                                    this_pd[this_different_key] = np.nan
                                else:
                                    pass
                        store.append(
                            this_hdf,
                            this_pd,
                            format="t",
                            data_columns=this_pd.columns,
                        )
                        store.close()

        summary_across_trials.append(df_summary)
        heading_direction_across_trials.append(angles)
        xy_across_trials.append(np.vstack((dX, dY)))
        if time_series_analysis:
            ts_across_trials.append(elapsed_time)
        else:
            ts_across_trials.append((ts - ts.min()).dt.total_seconds().values)
        if "agent_dX" in locals():
            agent_across_trials.append(agent_dXY_list)
            del agent_dX, agent_dY, df_agent
    trajectory_fig_path = this_file.parent / f"{experiment_id}_trajectory.png"
    save_curated_dataset(
        summary_file_path, ts_across_trials, pd.concat(summary_across_trials)
    )
    if plotting_trajectory == True and save_output == True:
        fig.savefig(trajectory_fig_path)
    return (
        heading_direction_across_trials,
        xy_across_trials,
        agent_across_trials,
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
        temperature_df = None
        print(f"temperature file not found")

    else:
        if isinstance(found_result, list):
            print(
                f"Multiple temperature files are detected. Have not figured out how to deal with this."
            )
            for this_file in found_result:
                temperature_df = load_temperature_data(this_file)
        else:
            temperature_df = load_temperature_data(found_result)
        if (
            "Celsius(°C)" in temperature_df.columns
        ):  # make the column name consistent with data from DL220 logger
            temperature_df.rename(
                columns={
                    "Celsius(°C)": "Temperature ˚C (ºC)",
                    "Humidity(%rh)": "Relative Humidity (%)",
                },
                inplace=True,
            )
    num_vr = 4
    scene_name = analysis_methods.get("experiment_name")
    ## here to load simulated agent's data
    for i in range(num_vr):
        if analysis_methods.get("export_fictrac_data_only",False)==False:
            if scene_name.lower() == "swarm" or scene_name.lower() == "band" or scene_name.lower()== "kannadi":
                agent_pattern = f"*SimulatedLocustsVR{i+1}*"
            elif scene_name.lower() == "choice":
                agent_pattern = "*Leader*"
            found_result = find_file(thisDir, agent_pattern)
            if found_result is None:
                return print(f"file with {agent_pattern} not found")
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
                            if type(this_config_file) == list:
                                this_config_file = this_config_file[0]
                            with open(this_config_file, "r") as f:
                                print(f"load trial conditions from file {this_config_file}")
                                trial_condition = json.loads(f.read())
                            num_object_on_scene = len(trial_condition["objects"])
                        else:
                            ## if not using config file then only one Unity object is present in a trial
                            trial_condition = trial_sequence["sequences"][idx]["parameters"]
                            num_object_on_scene = 1
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
                            for this_object in range(num_object_on_scene):
                                result, condition = read_agent_data(
                                    df[this_range],
                                    analysis_methods,
                                    trial_condition["objects"][this_object],
                                )

                                if num_object_on_scene > 1:
                                    # if isinstance(result, pd.DataFrame) == True:
                                    #     result.drop_duplicates(subset=["GameObjectPosZ"], keep='last',inplace=True)
                                    if this_object == 0:
                                        condition["duration"] = trial_sequence["sequences"][
                                            idx
                                        ][
                                            "duration"
                                        ]  # may need to add condition to exclude some kind of data from choice assay.
                                        condition_list.append(condition)
                                    if (
                                        (
                                            trial_condition["objects"][0]["type"]
                                            == trial_condition["objects"][1]["type"]
                                        )
                                        and (this_object == 1)
                                        and (trial_condition["objects"][0]["type"] != "")
                                    ):
                                        theta = np.radians(trial_condition["objects"][1]['mu']+270)
                                        # applying rotation matrix to rotate the coordinates
                                        # includes a minus because the radian circle is clockwise in Unity, so 45 degree should be used as -45 degree in the regular radian circle
                                        rot_matrix = np.array(
                                            [
                                                [np.cos(theta), -np.sin(theta)],
                                                [np.sin(theta), np.cos(theta)],
                                            ]
                                        )
                                        rXY = rot_matrix @ np.vstack(
                                            (
                                                result["GameObjectPosX"].values,
                                                result["GameObjectPosZ"].values,
                                            )
                                        )
                                        result["GameObjectPosX"] = rXY[1]
                                        result["GameObjectPosZ"] = rXY[0]
                                    result_list.append(result)
                                    if isinstance(result, pd.DataFrame) == True and (
                                        this_object == 1
                                    ):
                                        if len(result_list[0]) >= len(result_list[1]):
                                            matched_id = np.searchsorted(
                                                result_list[0]["Current Time"].values,
                                                result_list[1]["Current Time"].values,
                                            )
                                            if (
                                                matched_id.shape[0]
                                                == result_list[0].shape[0]
                                            ):
                                                pass
                                            elif matched_id[-1] >= result_list[0].shape[0]:
                                                result_list[0] = result_list[0].iloc[
                                                    matched_id[:-1], :
                                                ]
                                            else:
                                                result_list[0] = result_list[0].iloc[
                                                    matched_id, :
                                                ]
                                        elif len(result_list[0]) < len(result_list[1]):
                                            matched_id = np.searchsorted(
                                                result_list[1]["Current Time"].values,
                                                result_list[0]["Current Time"].values,
                                            )
                                            if (
                                                matched_id.shape[0]
                                                == result_list[1].shape[0]
                                            ):
                                                pass
                                            elif matched_id[-1] >= result_list[1].shape[0]:
                                                result_list[1] = result_list[1].iloc[
                                                    matched_id[:-1], :
                                                ]
                                            else:
                                                result_list[1] = result_list[1].iloc[
                                                    matched_id, :
                                                ]
                                        df_agent = reshape_multiagent_data(
                                            pd.concat(result_list), this_object
                                        )
                                        result_list = []
                                        result_list.append(df_agent)
                                    else:
                                        df_agent = None
                                else:
                                    result_list.append(result)
                        elif scene_name.lower()== "kannadi":
                            if num_object_on_scene==0:
                                df_agent = None
                                result_list.append(df_agent)
                                condition = trial_condition
                                condition['initial_position']=condition['vrConfigs'][i]['initialPosition']
                                condition['initial_orientation']=condition['vrConfigs'][i]['initialRotation']
                                if 'watchIndex' in condition['vrConfigs'][i]:
                                    condition['watchIndex']=condition['vrConfigs'][i]['watchIndex']
                                else:
                                    condition['watchIndex']=-1
                                del condition['vrConfigs']

                            elif "objects" in trial_condition:
                                for this_object in range(num_object_on_scene):
                                    result, condition = read_agent_data(
                                            found_result[
                                                int(idx * num_object_on_scene/2 + this_object)
                                            ],
                                            analysis_methods,
                                            trial_condition["objects"][this_object],
                                        )
                                    if isinstance(result, pd.DataFrame) == True:
                                        df_agent = reshape_multiagent_data(result, this_object)
                                    else:
                                        df_agent = None
                                    result_list.append(df_agent)                    
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
                                if (num_object_on_scene-this_object>1) and (num_object_on_scene>1):
                                    condition["duration"] = trial_sequence["sequences"][idx]["duration"]
                                    if type(trial_condition['closedLoopOrientation'])==bool:
                                        if trial_condition['closedLoopOrientation']==True:
                                            condition["rotation_gain"]=1.0
                                        else:
                                            condition["rotation_gain"]=0.0
                                    else:
                                        condition["rotation_gain"]=trial_condition['closedLoopOrientation']
                                    if type(trial_condition['closedLoopPosition'])==bool:
                                        if trial_condition['closedLoopPosition']==True:
                                            condition["translation_gain"]=1.0
                                        else:
                                            condition["translation_gain"]=0.0
                                    else:
                                        condition["translation_gain"]=trial_condition['closedLoopPosition']
                                     # may need to add condition to exclude some kind of data from choice assay.
                                    condition_list.append(condition)

                        condition["duration"] = trial_sequence["sequences"][idx][
                            "duration"
                        ]  # may need to add condition to exclude some kind of data from choice assay.
                        if type(trial_condition['closedLoopOrientation'])==bool:
                            if trial_condition['closedLoopOrientation']==True:
                                condition["rotation_gain"]=1.0
                            else:
                                condition["rotation_gain"]=0.0
                        else:
                            condition["rotation_gain"]=trial_condition['closedLoopOrientation']
                        if type(trial_condition['closedLoopPosition'])==bool:
                            if trial_condition['closedLoopPosition']==True:
                                condition["translation_gain"]=1.0
                            else:
                                condition["translation_gain"]=0.0
                        else:
                            condition["translation_gain"]=trial_condition['closedLoopPosition']

                        condition_list.append(condition)
                        if num_object_on_scene <= 1:
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
                            if type(this_config_file) == list:
                                this_config_file = this_config_file[0]
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
                            if type(trial_condition['closedLoopOrientation'])==bool:
                                if trial_condition['closedLoopOrientation']==True:
                                    condition["rotation_gain"]=1.0
                                else:
                                    condition["rotation_gain"]=0.0
                            else:
                                condition["rotation_gain"]=trial_condition['closedLoopOrientation']
                            if type(trial_condition['closedLoopPosition'])==bool:
                                if trial_condition['closedLoopPosition']==True:
                                    condition["translation_gain"]=1.0
                                else:
                                    condition["translation_gain"]=0.0
                            else:
                                condition["translation_gain"]=trial_condition['closedLoopPosition']
                            condition_list.append(condition)
                            if num_object_on_scene == 1:
                                conditions.append(condition_list[0])
                                df_simulated_animal.append(result_list[0])
                            else:
                                conditions.append(condition_list)
                                df_simulated_animal.append(result_list)
        else:
            df_simulated_animal=[]
            conditions=[]
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
                if scene_name.lower()== "kannadi":
                    found_result=[item for item in found_result if 'Clones' not in item.stem]
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
                            temperature_df,
                    )
                else:
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
                            temperature_df,
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
                    temperature_df,
                )


if __name__ == "__main__":
    #thisDir = r"D:\MatrexVR_2024_Data\RunData\20250523_143428"
    #thisDir = r"D:\MatrexVR_2024_3_Data\RunData\20250801_075938"
    thisDir = r"C:\Users\neuroLaptop\Documents\MatrexVR_2026_Data\RunData\20251114_173358"
    #thisDir = r"C:\Users\neuroLaptop\Documents\MatrexVR_2026_Data\RunData\20251114_185029"
    #thisDir = r"D:\MatrexVR_2024_Data\RunData\20250702_095817"
    #thisDir = r"D:\MatrexVR_2024_3_Data\RunData\20250709_155715"
    #thisDir = r"D:\MatrexVR_2024_Data\RunData\20250514_134255"
    #thisDir = r"D:\MatrexVR_2024_Data\RunData\20250605_120838"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    preprocess_matrex_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
