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
    df = pd.read_csv(this_file, sep=' ',header=None)

    # if 'velocities_all' in basepath:
    #     df = df.iloc[:,-4:-2]
    #     df = df.cumsum() * -1
    #s = "{} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(self.deltas[0], self.deltas[1],heading_direction,time.time(),trialNum,state,trial_label,pos1[0], pos1[1], pos2[0], pos2[1],posBaitx, posBaity)
    default_column_names = ["X","Y",'heading_direction',"ts","trial_id","state_type","trial_label","BaitX","BaitY",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2']    
    #default_column_names = ["BaitX","BaitY",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"x","y","trial_id","state_type","ts","trial_label"]
    #df.iloc[:,-5:].columns=default_column_names[-7:-2]
    df.columns=default_column_names[:df.shape[1]]

    ##The unit of raw data is in meters so we need to convert it to cm
    cols_to_convert=["X","Y",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"BaitX","BaitY"]
    df[cols_to_convert] = df[cols_to_convert]* 100
    #df_bait=df[["BaitX","BaitY", "state_type",'trial_id',"ts"]]
    #df_choices=df[['AgentX1', 'AgentY1', 'AgentX2', 'AgentY2', 'trial_id', 'state_type']]
    #df_XY=df[["x","y",'trial_id', 'state_type',"ts"]]
    #df_agent = pd.concat((pd.concat((df["BaitX"],df['AgentX1'], df['AgentX2']), ignore_index=True),pd.concat((df["BaitY"],df['AgentY1'], df['AgentY2']), ignore_index=True),pd.concat((df['trial_id'],df['trial_id'], df['trial_id']), ignore_index=True)),axis=1)
    #df_agent.columns=['X','Y',"state_type",'trial_id',"ts"]
    # df_summary["radial_distance"]
    trial_list=[]
    XY_list=[]
    bait_list=[]
    agent_list=[]
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
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 7), tight_layout=True)
        ax1.set_title("Bait state1")
        ax2.set_title("Choice state1")
        ax3.set_title("Bait state2")
        ax4.set_title("Choice state2")
    for this_trial in df['trial_id'].unique():
        pd_this_trial=df[df['trial_id'] == this_trial]
        X=np.cumsum(pd_this_trial['X'].to_numpy())
        Y=np.cumsum(pd_this_trial['Y'].to_numpy())        
        heading=pd_this_trial['heading_direction'].to_numpy()
        elapsed_time=pd_this_trial['ts'].to_numpy()-min(pd_this_trial['ts'].to_numpy())
        df_bait = pd_this_trial[['BaitX','BaitY','trial_id']]
        agent1=pd_this_trial[['AgentX1','AgentY1','trial_id']]
        agent1.columns=['X','Y','trial_id']
        agent1.loc[:,'agent_id']=list(np.ones(agent1.shape[0], dtype=int))
        agent2=pd_this_trial[['AgentX2','AgentY2','trial_id']]
        agent2.columns=['X','Y','trial_id']
        agent2.loc[:,'agent_id']=list(np.ones(agent2.shape[0], dtype=int)*2)
        state_type_summary= df[df['trial_id'] == this_trial]['state_type'].unique()[1]
        trial_label=df[df['trial_id'] == this_trial]['trial_label'].values[-1]##get trial label this way because in the first version, trial label is updated one row later 
        this_state_type=df[df['trial_id'] == this_trial]['state_type'].values
        if time_series_analysis:
            if analysis_methods.get("filtering_method") == "sg_filter":
                X = savgol_filter(X, 59, 3, axis=0)
                Y = savgol_filter(Y, 59, 3, axis=0)
            loss, dX, dY,mask= remove_unreliable_tracking(X, Y,analysis_methods)
            loss = 1 - loss
            if mask.shape[1]>0:
                heading.iloc[mask]=np.nan
            angles=heading
            dts =elapsed_time
            num_spatial_decision = len(angles)
        else:
            loss, X, Y,mask = remove_unreliable_tracking(X, Y,analysis_methods)
            elapsed_time=elapsed_time[1:][mask]
            df_bait=df_bait.iloc[1:,:][mask]
            agent1=agent1.iloc[1:,:][mask]
            agent2=agent2.iloc[1:,:][mask]
            loss = 1 - loss
            if len(X) == 0:
                continue
            rXY = np.vstack((X, Y))
            newindex = diskretize(list(rXY[0]), list(rXY[1]), BODY_LENGTH3)
            dX = rXY[0][newindex]
            dY = rXY[1][newindex]
            angles = np.arctan2(np.diff(dY), np.diff(dX)) ## in time series analysis, the angles should come from locustVR directly.
            angles = np.insert(angles, 0, np.nan)
            dts = elapsed_time[newindex]
            df_bait=df_bait.iloc[newindex,:]
            agent1=agent1.iloc[newindex,:]
            agent2=agent2.iloc[newindex,:]
            this_state_type=this_state_type[newindex]
            num_spatial_decision = len(angles) - 1
        df_agent=pd.concat([agent1,agent2],axis=0,ignore_index=True)
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
            'heading':angles,
            'ts':dts,
            'trial_id': this_trial,
            'state_type': this_state_type,
        })
        df_trial = pd.DataFrame({
            'fname': [fchop],
            'loss': loss,
            'trial_id': this_trial,
            'state_type':state_type_summary,
            'trial_label':trial_label,
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
            if state_type_summary==1:
                    ## if using plot instead of scatter plot
                ax1.plot(df_xy['X'][df_xy['state_type']==0].values, df_xy['Y'][df_xy['state_type']==0].values)
                ax1.scatter(df_bait['BaitX'][df_bait['BaitX']<10000].values, df_bait['BaitY'][df_bait['BaitY']<10000].values,c='k',s=1)
                ax2.plot(df_xy['X'][df_xy['state_type']==1].values, df_xy['Y'][df_xy['state_type']==1].values)
                ax2.scatter(df_agent['X'][df_agent['X']<10000].values, df_agent['Y'][df_agent['Y']<10000].values,c='k',s=1)
            else:
            ##blue is earlier colour and yellow is later colour
                ax3.plot(df_xy['X'][df_xy['state_type']==0].values, df_xy['Y'][df_xy['state_type']==0].values)
                ax3.scatter(df_bait['BaitX'][df_bait['BaitX']<10000].values, df_bait['BaitY'][df_bait['BaitY']<10000].values,c='k',s=1)
                ax4.plot(df_xy['X'][df_xy['state_type']==2].values, df_xy['Y'][df_xy['state_type']==2].values)
                ax4.scatter(df_agent['X'][df_agent['X']<10000].values, df_agent['Y'][df_agent['Y']<10000].values,c='k',s=1)

        trial_list.append(df_trial)
        XY_list.append(df_xy)
        bait_list.append(df_bait)
        agent_list.append(df_agent)
    df_summary=pd.concat(trial_list, ignore_index=True)
    df_curated=pd.concat(XY_list, ignore_index=True)
    df_bait=pd.concat(bait_list, ignore_index=True)
    df_agent=pd.concat(agent_list, ignore_index=True)
    # Use lock to prevent concurrent writes
    if save_output == True:
        with lock:
            file_list=[summary_file_path,curated_file_path,agent_file_path]
            data_frame_list=[df_summary,df_curated]
            df_agent_list=[df_bait,df_agent]
            agent_key_list=['bait','choices']            
            for i,this_file in enumerate(file_list):
                if this_file==agent_file_path:
                    store = pd.HDFStore(agent_file_path)
                    for this_agent, this_hdf_key in zip(df_agent_list, agent_key_list):
                        store.append(this_hdf_key, this_agent, format='t', data_columns=this_agent.columns)
                    store.close()
                else:
                    store = pd.HDFStore(this_file)
                    store.append("focal_animal", data_frame_list[i], format='t', data_columns=data_frame_list[i].columns)
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
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\locustVR\GN25003\20250612_1416_1749730564_2choice"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_files(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")