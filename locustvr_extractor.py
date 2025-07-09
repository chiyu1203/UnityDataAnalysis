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
from useful_tools import find_file,find_nearest
from data_cleaning import diskretize,remove_unreliable_tracking,euclidean_distance,remove_false_detection_heading,load_temperature_data

lock = Lock()
def extract_locustvr_dat(thisDir, analysis_methods):
    analysis_methods.update({"experiment_name": "locustvr"})
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
    this_file = find_file(thisDir, "data*.dat")
    if type(this_file) == str:
        this_file = Path(this_file)
    df = pd.read_csv(this_file, sep=' ',header=None)

    # if 'velocities_all' in basepath:
    #     df = df.iloc[:,-4:-2]
    #     df = df.cumsum() * -1
    #s = "{} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(self.deltas[0], self.deltas[1],heading_direction,time.time(),trialNum,state,trial_label,pos1[0], pos1[1], pos2[0], pos2[1],posBaitx, posBaity)
    
    ### dx and dy means derivative of x and y
    ### heading_direction comes from locustVR directly, which is estimated based on OpenCV blob detection
    ### ts is the timestamp in miliseconds
    ### trial_id is the trial number
    ### state_type is the state of the animal, 0 is pre-choice phase, 1 is choices type 1 , 2 is choices type 2
    ### trial_label indicates which type of stimuli is in choice type 1 and type 2
    ### baitX and baitY means the relative position of the bait to the focal animal during the prechoice phase
    ### AgentX1,X2 and AgentY1,Y2 means the position of the two agents in the trial

    default_column_names = ["dX","dY",'heading_direction',"ts","trial_id","state_type","trial_label",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"preChoice_relativeX","preChoice_relativeY"]    
    #default_column_names = ["preChoice_relativeX","preChoice_relativeY",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"x","y","trial_id","state_type","ts","trial_label"]
    #df.iloc[:,-5:].columns=default_column_names[-7:-2]
    df.columns=default_column_names[:df.shape[1]]

    ##The unit of raw data is in meters so we need to convert it to cm
    cols_to_convert=["dX","dY",'AgentX1', 'AgentY1', 'AgentX2', 'AgentY2',"preChoice_relativeX","preChoice_relativeY"]
    df[cols_to_convert] = df[cols_to_convert]* 100
    df = remove_false_detection_heading(df, angle_col='heading_direction', threshold_lower=3, threshold_upper=5.5, threshold_range=200)
    
    ## this can be used to recover the trajectory in the VR environment. However, it would make plotting trajectory more difficult so we rezero every position trial by trial
    # df['X']=df["dX"].cumsum()
    # df['Y']=df["dY"].cumsum()
    # df["preChoice_X"]=df["preChoice_relativeX"]+ df['X']
    # df["preChoice_Y"]=df["preChoice_relativeY"]+ df['Y']

    '''align temperature data with df'''
    [_,exp_date, exp_hour,_,_] = this_file.stem.split('_')
    exp_time=f"{exp_date}_{exp_hour}"
    exp_time_dt = pd.to_datetime(exp_time, format="%Y%m%d_%H%M")

    found_result = find_file(thisDir, "locustVR*.txt", "DL220THP*.csv")
    ## here to load temperature data
    if found_result is None:
        temperature_df = None
        print(f"temperature file not found")
    else:
        if isinstance(found_result, list):
            print(
                f"Multiple temperature files are detected. Have not figured out how to deal with this."
            )
            for this_f in found_result:
                temperature_df = load_temperature_data(this_f)
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


    if temperature_df is None:
        df["Temperature ˚C (ºC)"] = np.nan
        df["Relative Humidity (%)"] = np.nan
    else:
        start_time_tem = find_nearest(temperature_df.index, exp_time_dt)
        print(
            temperature_df[temperature_df.index == start_time_tem]
        )  # find the start of EL-USB data that might be used for further data analysis. For example, oversampling to align with existing data size
        tem_df_after_exp = temperature_df[temperature_df.index > start_time_tem]
        if len(tem_df_after_exp) > 0:
            frequency_milisecond = int(1000 / monitor_fps)
            tem_df_after_exp = tem_df_after_exp.resample(
                f"{frequency_milisecond}ms"
            ).interpolate()
            df["Temperature ˚C (ºC)"]=tem_df_after_exp["Temperature ˚C (ºC)"][:df.shape[0]].values
            df["Relative Humidity (%)"]=tem_df_after_exp["Relative Humidity (%)"][:df.shape[0]].values
        else:
            print("Unable to find the corresponding time stamp")
            df["Temperature ˚C (ºC)"] = np.nan
            df["Relative Humidity (%)"] = np.nan
        del temperature_df



    trial_list=[]
    XY_list=[]
    preChoice_list=[]
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
        ax1.set_title("preChoice state1")
        ax2.set_title("Choice state1")
        ax3.set_title("preChoice state2")
        ax4.set_title("Choice state2")
    for this_trial in df['trial_id'].unique():

        pd_this_trial=df[df['trial_id'] == this_trial]
        temperature_this_trial=pd_this_trial["Temperature ˚C (ºC)"].to_numpy()
        humidity_this_trial=pd_this_trial["Relative Humidity (%)"].to_numpy()
        X=np.cumsum(pd_this_trial['dX'].to_numpy())
        Y=np.cumsum(pd_this_trial['dY'].to_numpy())
        pd_this_trial['preChoice_relativeX']=pd_this_trial['preChoice_relativeX']+X
        pd_this_trial['preChoice_relativeY']=pd_this_trial['preChoice_relativeY']+Y
        df_preChoice = pd_this_trial[['preChoice_relativeX','preChoice_relativeY','trial_id']]
        
        # fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(
        #     nrows=4, ncols=2, figsize=(9,10), tight_layout=True
        # )
        # ax1.plot(np.arange(pd_this_trial.shape[0]),X)
        # ax2.plot(np.arange(pd_this_trial.shape[0]),Y)
        # ax3.plot(np.arange(pd_this_trial.shape[0]),pd_this_trial['preChoice_relativeX'])
        # ax4.plot(np.arange(pd_this_trial.shape[0]),pd_this_trial['preChoice_relativeY'])
        # ax5.scatter(pd_this_trial['preChoice_relativeX'],pd_this_trial['preChoice_relativeY'],c=np.arange(pd_this_trial.shape[0]),marker=".")
        # ax6.plot(np.arange(pd_this_trial.shape[0]),pd_this_trial['heading_direction'])
        # ax7.plot(np.arange(pd_this_trial.shape[0]),pd_this_trial['AgentX2'])## In the first trial, Agent2 is the constant speed one
        # ax8.plot(np.arange(pd_this_trial.shape[0]),pd_this_trial['AgentY2'])
        # # ax1.plot(np.arange(pd_this_trial.shape[0]),this_trajectory_X)
        # # ax2.plot(np.arange(pd_this_trial.shape[0]),this_trajectory_Y)
        # # ax3.scatter(this_trajectory_X,this_trajectory_Y,c=np.arange(pd_this_trial.shape[0]),marker=".")        
        # plt.show()
        
        
        ## this can be used to recover the trajectory in the VR environment. However, it would make plotting trajectory more difficult so we rezero every position trial by trial
        #X=pd_this_trial['X'].to_numpy()
        #Y=pd_this_trial['Y'].to_numpy()
        #df_preChoice = pd_this_trial[['preChoice_X','preChoice_Y','trial_id']]
        df_preChoice.columns=['X','Y','trial_id']
        agent1=pd_this_trial[['AgentX1','AgentY1','trial_id']]
        agent1.columns=['X','Y','trial_id']
        agent1.loc[:,'agent_id']=list(np.ones(agent1.shape[0], dtype=int))
        agent2=pd_this_trial[['AgentX2','AgentY2','trial_id']]
        agent2.columns=['X','Y','trial_id']
        agent2.loc[:,'agent_id']=list(np.ones(agent2.shape[0], dtype=int)*2)
        heading=pd_this_trial['heading_direction'].to_numpy()
        elapsed_time=pd_this_trial['ts'].to_numpy()-min(pd_this_trial['ts'].to_numpy())
        if len(df[df['trial_id'] == this_trial]['state_type'].unique())==1:
            continue # if a trial only has 1 state type, that means that trial never enter the actual choice phase, skip this trial
        else:
            state_type_summary= df[df['trial_id'] == this_trial]['state_type'].unique()[1]
        label_split=df[df['trial_id'] == this_trial]['trial_label'].values[-1].split("_")##get trial label this way because in the first version, trial label is updated one row later 
        trial_label=f"{label_split[1]}_{label_split[2]}_{label_split[3]}"##remove T information because that is redundant 
        this_state_type=df[df['trial_id'] == this_trial]['state_type'].values
        if time_series_analysis:
            if analysis_methods.get("filtering_method") == "sg_filter":
                X = savgol_filter(X, 59, 3, axis=0)
                Y = savgol_filter(Y, 59, 3, axis=0)
            loss, curated_X, curated_Y,mask= remove_unreliable_tracking(X, Y,analysis_methods)
            loss = 1 - loss
            if mask.shape[1]>0:
                heading[mask]=np.nan
            angles=heading
            dts =elapsed_time
            num_spatial_decision = len(angles)
        else:
            loss, X, Y,mask = remove_unreliable_tracking(X, Y,analysis_methods)
            elapsed_time=elapsed_time[1:][mask]
            df_preChoice=df_preChoice.iloc[1:,:][mask]
            agent1=agent1.iloc[1:,:][mask]
            agent2=agent2.iloc[1:,:][mask]
            loss = 1 - loss
            if len(X) == 0:
                continue
            rXY = np.vstack((X, Y))
            newindex = diskretize(list(rXY[0]), list(rXY[1]), BODY_LENGTH3)
            curated_X = rXY[0][newindex]
            curated_Y = rXY[1][newindex]
            angles = np.arctan2(np.diff(curated_Y), np.diff(curated_X)) ## in time series analysis, the angles should come from locustVR directly.
            angles = np.insert(angles, 0, np.nan)
            dts = elapsed_time[newindex]
            df_preChoice=df_preChoice.iloc[newindex,:]
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
            np.nansum(euclidean_distance(X,Y))
            if time_series_analysis
            else len(curated_X) * BODY_LENGTH3
        )
        chop = str(thisDir).split('\\')[-1].split('_')[:2]
        fchop = '_'.join(chop)
        df_xy = pd.DataFrame({
            'X': curated_X,
            'Y': curated_Y,
            'heading':angles,
            'ts':dts,
            'trial_id': this_trial,
            'state_type': this_state_type,
        })
        df_trial = pd.DataFrame({
            'fname': [thisDir],
            'loss': loss,
            'trial_id': this_trial,
            'state_type':state_type_summary,
            'trial_label':trial_label,
            "groups": [growth_condition],
            'score': meanAngle,
            'vector': meanVector,
            'variance': std,
            'distX': curated_X[-1],
            'distTotal': tdist,
            'sin': VecSin,
            'cos': VecCos,
            "temperature":temperature_this_trial[0],
            "humidity":humidity_this_trial[0]
        })
        if plotting_trajectory == True:
            if state_type_summary==1:
                    ## if using plot instead of scatter plot
                ax1.plot(df_xy['X'][df_xy['state_type']==0].values, df_xy['Y'][df_xy['state_type']==0].values)
                ax1.scatter(df_preChoice['X'][df_preChoice['X']<10000].values, df_preChoice['Y'][df_preChoice['Y']<10000].values,c='k',s=1)
                ax2.plot(df_xy['X'][df_xy['state_type']==1].values, df_xy['Y'][df_xy['state_type']==1].values)
                ax2.scatter(df_agent['X'][df_agent['X']<10000].values, df_agent['Y'][df_agent['Y']<10000].values,c='k',s=1)
            else:
            ##blue is earlier colour and yellow is later colour
                ax3.plot(df_xy['X'][df_xy['state_type']==0].values, df_xy['Y'][df_xy['state_type']==0].values)
                ax3.scatter(df_preChoice['X'][df_preChoice['X']<10000].values, df_preChoice['Y'][df_preChoice['Y']<10000].values,c='k',s=1)
                ax4.plot(df_xy['X'][df_xy['state_type']==2].values, df_xy['Y'][df_xy['state_type']==2].values)
                ax4.scatter(df_agent['X'][df_agent['X']<10000].values, df_agent['Y'][df_agent['Y']<10000].values,c='k',s=1)

        trial_list.append(df_trial)
        XY_list.append(df_xy)
        preChoice_list.append(df_preChoice)
        agent_list.append(df_agent)
    if len(trial_list)==0:
        return print(f'Unable to process file from {thisDir}. Probably because the animal has never entered a choice phase')
    df_summary=pd.concat(trial_list, ignore_index=True)
    df_curated=pd.concat(XY_list, ignore_index=True)
    df_preChoice=pd.concat(preChoice_list, ignore_index=True)
    df_agent=pd.concat(agent_list, ignore_index=True)
    # Use lock to prevent concurrent writes
    if save_output == True:
        with lock:
            file_list=[summary_file_path,curated_file_path,agent_file_path]
            data_frame_list=[df_summary,df_curated]
            df_agent_list=[df_preChoice,df_agent]
            agent_key_list=['preChoice','Choices']            
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
    extract_locustvr_dat(thisDir,analysis_methods)


if __name__ == "__main__":
    #thisDir = r"Z:\DATA\experiment_trackball_Optomotor\locustVR\GN25003\20250612_1416_1749730564_2choice"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\locustVR\GN25012\20250625\choices\session1"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_files(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")