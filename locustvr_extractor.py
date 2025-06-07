import os,time,json,sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import atan2
from threading import Lock
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file
from data_cleaning import diskretize,remove_unreliable_tracking,euclidean_distance

lock = Lock()
def process_file(this_file, analysis_methods,count):
    monitor_fps = analysis_methods.get("monitor_fps")
    plotting_trajectory = analysis_methods.get("plotting_trajectory", False)
    save_output = analysis_methods.get("save_output", False)
    overwrite_curated_dataset = analysis_methods.get("overwrite_curated_dataset", False)
    time_series_analysis = analysis_methods.get("time_series_analysis", False)
    experiment_name = analysis_methods.get("experiment_name")
    growth_condition = analysis_methods.get("growth_condition")
    BODY_LENGTH3 = (
        analysis_methods.get("body_length", 4) * 3
    )
    df = pd.read_csv(this_file, sep=' ', header=0)

    # if 'velocities_all' in basepath:
    #     df = df.iloc[:,-4:-2]
    #     df = df.cumsum() * -1
    #s = "{} {} {} {} {} {} {} {} {} {} {} {}\n".format(self.deltas[0], self.deltas[1],time.time(),trialNum,state,pos1[0], pos1[1], pos1_variable, pos2[0], pos2[1],pos2_variable,posBaitx, posBaity)
    #default_column_names = ["X","Y","trial_id","trial_type","ts","variable1","variable2"]
    default_column_names = ["BaitX","BaitY",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"x","y","trial_id","trial_type","ts","variable1","variable2"]
    #df.iloc[:,-5:].columns=default_column_names[-7:-2]
    df.columns=default_column_names[:df.shape[1]]

    ##The unit of raw data is in meters so we need to convert it to cm
    cols_to_convert=["BaitX","BaitY",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"x","y"]
    df[cols_to_convert] = df[cols_to_convert]* 100
    df_bait=df[["BaitX","BaitY", 'trial_id',"trial_type","ts"]]
    #df_choices=df[['AgentX1', 'AgentY1', 'AgentX2', 'AgentY2', 'trial_id', 'trial_type']]
    #df_XY=df[["x","y",'trial_id', 'trial_type',"ts"]]
    df_agent = pd.concat([pd.concat([df['AgentX1'], df['AgentX2']], ignore_index=True),pd.concat([df['AgentY1'], df['AgentY2']], ignore_index=True),pd.concat([df['trial_id'], df['trial_id']], ignore_index=True),pd.concat([df['trial_type'], df['trial_type']], ignore_index=True),pd.concat([df['ts'], df['ts']], ignore_index=True)],axis=1)
    df_agent.columns=['x','y','trial_id', 'trial_type',"ts"]
    # df_summary["radial_distance"]
    trial_list=[]
    XY_list=[]
    file_suffix = "_full" if time_series_analysis else ""
    curated_file_path = this_file.parent / f"XY{file_suffix}.h5"
    summary_file_path = this_file.parent / f"summary{file_suffix}.h5"
    agent_file_path = this_file.parent / f"agent{file_suffix}.h5"
    trajectory_fig_path = this_file.parent / f"trajectory{file_suffix}.png"
    if overwrite_curated_dataset:
        summary_file_path.unlink(missing_ok=True)
        curated_file_path.unlink(missing_ok=True)
        agent_file_path.unlink(missing_ok=True)
    if plotting_trajectory:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), tight_layout=True)
        ax1.set_title("ISI")
        ax2.set_title("Trial")
    for this_trial in df['trial_id'].unique():
        X=np.cumsum(df[df['trial_id'] == this_trial]['x'].to_numpy())
        Y=np.cumsum(df[df['trial_id'] == this_trial]['y'].to_numpy())
        ts= df[df['trial_id'] == this_trial]['ts'].to_numpy()
        trial_type= df[df['trial_id'] == this_trial]['trial_type'][0]
        if time_series_analysis:
            #elapsed_time = ts - ts.min()
            if analysis_methods.get("filtering_method") == "sg_filter":
                X = savgol_filter(X, 59, 3, axis=0)
                Y = savgol_filter(Y, 59, 3, axis=0)
            loss, dX, dY,dts = remove_unreliable_tracking(X, Y,analysis_methods,ts)
            loss = 1 - loss
            #num_spatial_decision = len(angles) The angle here should come from locustVR directly
        else:
            loss, X, Y,ts = remove_unreliable_tracking(X, Y,analysis_methods,ts)
            loss = 1 - loss

            if len(X) == 0:
                continue
            rXY = np.vstack((X, Y))
            newindex = diskretize(list(rXY[0]), list(rXY[1]), BODY_LENGTH3)
            dX = rXY[0][newindex]
            dY = rXY[1][newindex]
            dts = ts[newindex]
        angles = np.arctan2(np.diff(dY), np.diff(dX)) ## in time series analysis, the angles should come from locustVR directly.
        angles = np.insert(angles, 0, np.nan)
        num_spatial_decision = len(angles) - 1
        c = np.cos(angles)
        s = np.sin(angles)
        if len(angles) == 0:
            xm = ym = meanAngle = meanVector = VecSin = VecCos = np.nan
        else:
            xm = np.nansum(c) / len(angles)
            ym = np.nansum(s) / len(angles)
            meanAngle = atan2(ym, xm)
            meanVector = np.sqrt(np.square(np.sum(c)) + np.square(np.sum(s))) / num_spatial_decision

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
        )
        chop = str(this_file.parent).split('\\')[-1].split('_')[:2]
        fchop = '_'.join(chop)
        df_xy = pd.DataFrame({
            'X': dX,
            'Y': dY,
            'ts':dts,
            'trial_id': this_trial,
        })
        df_trial = pd.DataFrame({
            'fname': [fchop],
            'loss': loss,
            'trial_id': this_trial,
            'trial_type':trial_type,
            "groups": [growth_condition],
            'score': meanAngle,
            'vector': meanVector,
            'variance': std,
            'distX': dX[-1],
            'distTotal': tdist,
            'sin': VecSin,
            'cos': VecCos,
        })
        if plotting_trajectory == True:
            if trial_type!=0:
                    ## if using plot instead of scatter plot
                ax2.plot(dX, dY)
            else:
            ##blue is earlier colour and yellow is later colour
                ax1.plot(dX, dY)
                    # ax1.scatter(
                    #     dX,
                    #     dY,
                    #     c=np.arange(len(dY)),
                    #     marker=".",
                    # )
                    # ax2.scatter(
                    #     dX,
                    #     dY,
                    #     c=np.arange(len(dY)),
                    #     marker=".",
                    # )
                    # if "agent_dX" in locals():
                    #     for j in range(len(df_agent_list)):
                    #         this_pd = df_agent_list[j]
                    #         ax2.plot(
                    #             this_pd["X"].values,
                    #             this_pd["Y"].values,
                    #             c="k",
                    #             linewidth=1,
                    #         )
        trial_list.append(df_trial)
        XY_list.append(df_xy)
        # bait_list.append(df_trial)
        # agent_list.append(df_trial)
    # if time_series_analysis:
    #     df_xy=df_xy
    # else:
    #     df_xy
    df_summary=pd.concat(trial_list, ignore_index=True)
    df_curated=pd.concat(XY_list, ignore_index=True)
    # Use lock to prevent concurrent writes
    if save_output == True:
        with lock:
            store = pd.HDFStore(curated_file_path)
            store.append('name_of_frame', df_curated, format='t', data_columns=df_curated.columns)
            store.close()
            store = pd.HDFStore(summary_file_path)
            store.append('name_of_frame', df_summary, format='t', data_columns=df_summary.columns)
            store.close()
    
    if plotting_trajectory == True and save_output == True:
        fig.savefig(trajectory_fig_path)
    return print(f'finished preprocessing step for {fchop} ')
def load_files(thisDir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    found_result = find_file(thisDir, "*.dat")
    if isinstance(found_result, list):
        for count, f in enumerate(found_result):
            process_file(f,analysis_methods,count)
    elif len(found_result.stem) > 0:
        process_file(found_result,analysis_methods,"")


if __name__ == "__main__":
    thisDir = r"Z:\DATA\experiment_locustVR\Data\20250606_1428_1749212917_2choice"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_files(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")