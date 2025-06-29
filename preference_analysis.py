from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import seaborn as sns
import math
import pandas as pd
import json,pickle

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
def plot_violin_across_parameters(ax,raw_data,thresholds,data_color='k'):
    means= np.nanmean(raw_data, axis=1)
    nested_list=raw_data.tolist()
    data = [sorted([x for x in sublist if not math.isnan(x)]) for sublist in nested_list]
    parts = ax.violinplot(
    data, thresholds, showmeans=False, showmedians=False,
    showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(data_color)
        pc.set_edgecolor(data_color)
        pc.set_alpha(0.3)
    quartile1 = [np.percentile(d, 25) for d in data]
    medians   = [np.percentile(d, 50) for d in data]
    quartile3 = [np.percentile(d, 75) for d in data]
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    #inds = thresholds
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.scatter(thresholds, medians, marker='o', color='white', s=20, zorder=3)
    ax.scatter(thresholds, means, marker='o', color='grey', s=20, zorder=3)
    ax.vlines(thresholds, quartile1, quartile3, color=data_color, linestyle='-', lw=5)
    ax.vlines(thresholds, whiskers_min, whiskers_max, color=data_color, linestyle='-', lw=1)
    return ax
def plot_scatter_across_parameters(ax,raw_data,thresholds,data_color='k'):
    xaxis_points=np.repeat(thresholds,raw_data.shape[1]).reshape(len(thresholds),raw_data.shape[1])
    ax.scatter(xaxis_points,raw_data,alpha=0.2,c=data_color)
    mean_index=np.mean(xaxis_points,axis=1)
    mean_response=np.nanmean(raw_data,axis=1)
    _, median_response, _ = np.nanpercentile(raw_data, [25, 50, 75], axis=1)
    sem_response=np.nanstd(raw_data, axis=1, ddof=1) / np.sqrt(raw_data.shape[1])
    ax.errorbar(
        mean_index,
        mean_response,
        yerr=sem_response,
        c=data_color,
        fmt="o",
        elinewidth=1,
        capsize=2,
    )
    ax.scatter(mean_index,median_response,s=200,marker='x',c=data_color)
    return ax
def plot_jitter_across_parameters(ax,raw_data,thresholds,data_color='k'):
    xaxis_points=np.repeat(thresholds,raw_data.shape[1])
    test=np.column_stack((np.transpose(np.hstack(raw_data)),np.transpose(xaxis_points)))
    data = pd.DataFrame(test,columns=['value','ROI size'])
    sns.stripplot(data=data,x='ROI size',y='value',ax=ax,jitter=True, color=data_color, alpha=0.2)
    return ax

def plot_relative_pos_distribution(relative_pos_of_interest,trial_type_of_interest,distance_threshold_for_plotting,analysis_methods,this_vr='all'):
    save_output= analysis_methods.get("save_output")
    xlimit=(0,distance_threshold_for_plotting)
    ylimit=(-1*distance_threshold_for_plotting,distance_threshold_for_plotting)
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(18, 6),tight_layout=True,sharex=True, sharey=True)
    if len(relative_pos_of_interest)==1:
        ax1.hist2d(relative_pos_of_interest[0]['x'].values,relative_pos_of_interest[0]['y'].values,bins=50)
    else:
        ax1.hist2d(np.hstack((relative_pos_of_interest[0]['x'].values,relative_pos_of_interest[1]['x'].values)),np.hstack((relative_pos_of_interest[0]['y'].values,relative_pos_of_interest[1]['y'].values)),bins=50)
        ax2.hist2d(np.hstack((relative_pos_of_interest[0]['x'].values,relative_pos_of_interest[1]['x'].values)),np.hstack((relative_pos_of_interest[0]['y'].values,relative_pos_of_interest[1]['y'].values*-1)),bins=50)
        ax2.set(yticks=[-1*distance_threshold_for_plotting,-5,0,5,distance_threshold_for_plotting],xticks=[0,5,distance_threshold_for_plotting],xlim=xlimit,ylim=ylimit,title='agent preference',adjustable='box', aspect='equal')
    ax1.set(
    yticks=[-1*distance_threshold_for_plotting,-5,0,5,distance_threshold_for_plotting],
    xticks=[0,5,distance_threshold_for_plotting],
    xlim=xlimit,ylim=ylimit,title='spatial preference',adjustable='box', aspect='equal')
    if len(relative_pos_of_interest)==1:
        fig_name=f"heatmap2D_{this_vr}_{trial_type_of_interest[0]}_distance_threshold_{distance_threshold_for_plotting}.png"
    else:
        fig_name=f"heatmap2D_{this_vr}_{trial_type_of_interest[0]}_and_{trial_type_of_interest[1]}_distance_threshold_{distance_threshold_for_plotting}.png"
    if save_output==True:
        fig.savefig(fig_name)
    plt.show()
def plot_preference_index(left_right_preference_across_animals,exp_con_preference_across_animals,trial_type_of_interest,analysis_methods,thresholds=[4,5,6,7,8],this_vr='all'):
    save_output= analysis_methods.get("save_output")
    frequency_based_preference_index=analysis_methods.get("frequency_based_preference_index")
    if analysis_methods.get("exclude_extreme_index"):
        annotation="_no_extreme"
    else:
        annotation=""
    if frequency_based_preference_index:
        preference_index_type='_frequency_based'
    else:
        preference_index_type='_time_based'
    violin_plot=False
    jitter_plot=False
    object_of_interest=trial_type_of_interest[0].split("_x_")
    
    if len(object_of_interest)==1 and len(trial_type_of_interest)==2:
        fig_name=f"preference_index_VR{this_vr}_homogenous_trials_multiple_condition_top_{trial_type_of_interest[1]}_bottom_{trial_type_of_interest[0]}"
        data_color='k'
    elif len(object_of_interest)==2 and len(trial_type_of_interest)==2:
        fig_name=f"preference_index_VR{this_vr}_heterogenous_trials_multiple_condition_top_{object_of_interest[1]}_bottom_{object_of_interest[0]}"
        data_color='b'
    elif len(object_of_interest)==1 and len(trial_type_of_interest)==1:
        fig_name=f"preference_index_VR{this_vr}_homogenous_trials_single_condition_object_type_{object_of_interest[0]}"
        data_color='k'
    elif len(object_of_interest)==2 and len(trial_type_of_interest)==1:
        fig_name=f"preference_index_VR{this_vr}_heterogenous_trials_single_condition_top_{object_of_interest[1]}_bottom_{object_of_interest[0]}"
        data_color='b'
    else:
        print("error. trial type or object type used in the analysis should be no more than 2")
    preference_plot, (ax1,ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 4), tight_layout=True
    )

    if violin_plot:
        ax1=plot_violin_across_parameters(ax1,left_right_preference_across_animals,thresholds,data_color=data_color)
        ax2=plot_violin_across_parameters(ax2,exp_con_preference_across_animals,thresholds,data_color='r')
        fig_type=f"_violin"
    elif jitter_plot:
        ax1=plot_jitter_across_parameters(ax1,left_right_preference_across_animals,thresholds,data_color=data_color)
        ax2=plot_jitter_across_parameters(ax2,exp_con_preference_across_animals,thresholds,data_color='r')
        fig_type=f"_jitter"
    else:
        ax1=plot_scatter_across_parameters(ax1,left_right_preference_across_animals,thresholds,data_color=data_color)
        if 'homogenous_trials' in fig_name:
            ax2=plot_scatter_across_parameters(ax2,exp_con_preference_across_animals,thresholds,data_color='k')
        else:
            ax2=plot_scatter_across_parameters(ax2,exp_con_preference_across_animals,thresholds,data_color='r')
        fig_type=f"_scatter"
    if jitter_plot ==False:
        ax1.set(
            ylabel="(postive means prefer left)",
            yticks=[-1,0,1],
            xticks=thresholds,
            xlabel="ROI size (cm)",
            title="spatial preference",
            xlim=(min(thresholds)-1,max(thresholds)+1)
        )
        ax2.set(
            #ylabel="(positive means prefer exp)",
            yticks=[-1,0,1],
            xticks=thresholds,
            xlabel="ROI size (cm)",
            title="visual preference",
            xlim=(min(thresholds)-1,max(thresholds)+1)
        )
    if save_output==True:
        preference_plot.savefig(f"{fig_name}{fig_type}{preference_index_type}{annotation}.png")
        preference_plot.savefig(f"{fig_name}{fig_type}{preference_index_type}{annotation}.svg")
    plt.show()
def plot_epochs_time(epochs_exp,epochs_con,epochs_L,epochs_R,analysis_methods,fig_name,data_color,thresholds=[4,5,6,7,8],this_vr='all'):
    ##try using seaborn to plot the data next time
    save_output=analysis_methods.get("save_output")
    experiment_name=analysis_methods.get("experiment_name")
    camera_fps=analysis_methods.get("camera_fps")
    frequency_based_preference_index=analysis_methods.get("frequency_based_preference_index")
    if analysis_methods.get("exclude_extreme_index"):
        annotation="_no_extreme"
    else:
        annotation=""
    if frequency_based_preference_index:
        unit='(count)'
        camera_fps=1
        limits=[35,40,45,50,55]
        fig_type='_frequency_based'
    elif experiment_name=='band':
        unit='(sec)'
        limits=[100,200,300,400,500]
        fig_type='_time_based'
    else:
        unit='(sec)'
        limits=[25,30,35,40,45]
        fig_type='_time_based'
    num_subplots=len(thresholds)
    scatterplots, axes = plt.subplots(
        nrows=1, ncols=num_subplots*2, figsize=(40, 4), tight_layout=True
    )
    if 'homogenous_trials' in fig_name:
        fig_color='k'
    else:
        fig_color='b'

    #limits=[15,20,25,30,35]
    
    for i in range(epochs_exp.shape[0]):
        axes[i].scatter(epochs_con[i,:]/camera_fps,epochs_exp[i,:]/camera_fps,c=fig_color)
        axes[i].set(
            xlabel=f"con {unit}",
            ylabel=f"exp {unit}",
            title=f"threshold {thresholds[i]} cm",
            xlim=(0,limits[i]),ylim=(0,limits[i]),
            xticks=[0,limits[i]//2,limits[i]],yticks=[0,limits[i]//2,limits[i]],
        )
        axes[i].plot([0, 1], [0, 1], transform=axes[i].transAxes, color='gray', linestyle='--')
        axes[i+num_subplots].scatter(epochs_L[i,:]/camera_fps,epochs_R[i,:]/camera_fps,c=data_color)
        axes[i+num_subplots].set(
            xlabel=f"L {unit}",
            ylabel=f"R {unit}",
            title=f"threshold {thresholds[i]} cm",
            xlim=(0,limits[i]),ylim=(0,limits[i]),
            xticks=[0,limits[i]//2,limits[i]],yticks=[0,limits[i]//2,limits[i]],
        )
        axes[i+num_subplots].plot([0, 1], [0, 1], transform=axes[i+num_subplots].transAxes, color='gray', linestyle='--')
    if save_output==True:
        scatterplots.savefig(f"{fig_name}{fig_type}{annotation}.png")
        scatterplots.savefig(f"{fig_name}{fig_type}{annotation}.svg")

def calculate_preference_index(relative_pos_all_animals,trial_type_of_interest,analysis_methods,thresholds=[4,5,6,7,8],this_vr='all'):
    frequency_based_preference_index=analysis_methods.get("frequency_based_preference_index")
    exclude_extreme_index=analysis_methods.get("exclude_extreme_index",False)
    spatial_preference_animals=np.zeros((len(thresholds),len(relative_pos_all_animals),relative_pos_all_animals[0]['type'].unique().shape[0]))
    spatial_preference_animals[:]=np.nan
    epochs_forL_all_animals_homo=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forL_all_animals_homo[:]=np.nan
    epochs_forR_all_animals_homo=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forR_all_animals_homo[:]=np.nan
    epochs_forL_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forL_all_animals_hetero[:]=np.nan
    epochs_forR_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forR_all_animals_hetero[:]=np.nan
    epochs_exp_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_exp_all_animals_hetero[:]=np.nan
    epochs_con_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_con_all_animals_hetero[:]=np.nan
    object_of_interest=trial_type_of_interest[0].split("_x_")
    if len(object_of_interest)==1 and len(trial_type_of_interest)==2:
        fig_name=f"epochs_VR{this_vr}_homegenous_trials_multiple_condition_{trial_type_of_interest[0]}_and_{trial_type_of_interest[1]}"
        data_color='k'
    elif len(object_of_interest)==2 and len(trial_type_of_interest)==2:
        print(f"assign epoch con is for {object_of_interest[0]} and epochs exp is for {object_of_interest[1]}")
        fig_name=f"epochs_VR{this_vr}_heterogenous_trials_multiple_condition_con_{object_of_interest[0]}_exp_{object_of_interest[1]}"
        data_color='b'
    elif len(object_of_interest)==1 and len(trial_type_of_interest)==1:
        fig_name=f"epochs_VR{this_vr}_homegenous_trials_single_condition_object_type_{object_of_interest[0]}"
        data_color='k'
    elif len(object_of_interest)==2 and len(trial_type_of_interest)==1:
        fig_name=f"epochs_VR{this_vr}_heterogenous_trials_single_condition_object_type_{object_of_interest[0]}_and_{object_of_interest[1]}"
        data_color='b'
    else:
        print("error. trial type or object type used in the analysis should be no more than 2")
    for i,relative_pos_this_animal in enumerate(relative_pos_all_animals):
        trial_type_list=sorted(relative_pos_this_animal['type'].unique(), key=len)
        if len(trial_type_list)<len(pd.concat(relative_pos_all_animals,ignore_index=True)['type'].unique()):
            print(f"animal {i} only three follow epochs from {len(trial_type_list)} trial types")
            continue
        homo_no=0
        hetero_no=0
        epochs_forL_this_animal_homo=np.zeros((len(thresholds),2))
        epochs_forR_this_animal_homo=np.zeros((len(thresholds),2))
        epochs_forL_this_animal_hetero=np.zeros((len(thresholds),2))
        epochs_forR_this_animal_hetero=np.zeros((len(thresholds),2))
        epochs_exp_this_animal_hetero=np.zeros((len(thresholds),2))
        epochs_con_this_animal_hetero=np.zeros((len(thresholds),2))
        relative_pos_this_animal['distance']=LA.norm(np.vstack((relative_pos_this_animal["x"].values,relative_pos_this_animal["y"].values)),axis=0)
        #for key,grp in relative_pos_this_animal.groupby('type'):
        for key in trial_type_list:
            grp=relative_pos_this_animal[relative_pos_this_animal['type']==key]
            #Note1: start to do analysis based on the functionality of groupby and sorted
            #The first trial type to analyse is based on alphabetical order, and then based on length of the string
            #relative_distance=grp['distance'].values
            epochs_forL_array=np.zeros((len(thresholds)))
            epochs_forR_array=np.zeros((len(thresholds)))
            epochs_exp_array=np.zeros((len(thresholds)))
            epochs_con_array=np.zeros((len(thresholds)))
            for j,this_threshold in enumerate(thresholds):
                ## when entering ROI happens, based on the agent_id to know whether it is enter L or R
                ##return agent ID when relative distance is within the threshold
                
                ### try using numpy mask in the future
                # enter_roi=(grp.loc[:,'distance']<this_threshold).to_numpy()
                # agent_id=grp.loc[1:,'agent_id'].to_numpy()
                # agent_id[1:][enter_roi[1:] & (~enter_roi[:-1])]
                
                ### now use this for readability of the code. The warning comes from the fact that grp is a truncated dataframe
                grp.loc[:,'enter_roi']=grp.loc[:,'distance']<this_threshold
                transitions_toL = (grp['enter_roi'].shift(1) == False) & (grp['enter_roi'] == True) & (grp['agent_id'] == 1)
                visit_frequency_L=transitions_toL.sum()
                transitions_toR = (grp['enter_roi'].shift(1) == False) & (grp['enter_roi'] == True) & (grp['agent_id'] == 0)
                visit_frequency_R=transitions_toR.sum()
                if frequency_based_preference_index:
                    epochs_forL=visit_frequency_L
                    epochs_forR=visit_frequency_R
                else:
                    ##sum agent ID to get epochs for the left object because agent ID is 0 for the right object and 1 for the left object
                    #epochs_forL=np.sum(grp[relative_distance<this_threshold]["agent_id"].values)### try using .loc[row_index, column_name]=value instead
                    epochs_forL=grp.loc[grp['enter_roi']==True,'agent_id'].sum()
                    ##epochs for the R object comes from the rest of element in the array
                    epochs_forR=grp.loc[grp['enter_roi']==True].shape[0]-epochs_forL

                ##assign epochs_exp or epochs_con based on the trial type, if the trial type is not of interest, assign NaN
                if len(trial_type_of_interest)==2:
                    if key==trial_type_of_interest[1]:
                        epochs_exp=epochs_forL
                        epochs_con=epochs_forR
                    elif key==trial_type_of_interest[0]:
                        epochs_exp=epochs_forR
                        epochs_con=epochs_forL
                    else:
                        ## refer the value to nan if the key is not this type. This seperate no exploration from wrong trial type
                        (epochs_con,epochs_exp,epochs_forL,epochs_forR)=(np.nan,np.nan,np.nan,np.nan)
                    
                elif len(trial_type_of_interest)==1:
                    if len(object_of_interest)==1 and key==object_of_interest[0]:
                        epochs_exp=epochs_forR
                        epochs_con=epochs_forL
                    elif len(object_of_interest)==2 and key==trial_type_of_interest[0]:
                        epochs_exp=epochs_forR
                        epochs_con=epochs_forL
                    else:
                        (epochs_con,epochs_exp,epochs_forL,epochs_forR)=(np.nan,np.nan,np.nan,np.nan)
                epochs_forL_array[j]=epochs_forL
                epochs_forR_array[j]=epochs_forR
                epochs_exp_array[j]=epochs_exp
                epochs_con_array[j]=epochs_con
                
            if len(key.split("_x_"))==1:
                epochs_forL_this_animal_homo[:,homo_no]=epochs_forL_array
                epochs_forR_this_animal_homo[:,homo_no]=epochs_forR_array
                homo_no=homo_no+1
            elif len(key.split("_x_"))==2:
                epochs_forL_this_animal_hetero[:,hetero_no]=epochs_forL_array
                epochs_forR_this_animal_hetero[:,hetero_no]=epochs_forR_array
                epochs_exp_this_animal_hetero[:,hetero_no]=epochs_exp_array
                epochs_con_this_animal_hetero[:,hetero_no]=epochs_con_array
                hetero_no=hetero_no+1
            else:
                Warning("unknown trial")
        epochs_forL_all_animals_homo[:,i]=np.nansum(epochs_forL_this_animal_homo,axis=1)
        epochs_forR_all_animals_homo[:,i]=np.nansum(epochs_forR_this_animal_homo,axis=1)
        epochs_forL_all_animals_hetero[:,i]=np.nansum(epochs_forL_this_animal_hetero,axis=1)
        epochs_forR_all_animals_hetero[:,i]=np.nansum(epochs_forR_this_animal_hetero,axis=1)
        epochs_exp_all_animals_hetero[:,i]=np.nansum(epochs_exp_this_animal_hetero,axis=1)
        epochs_con_all_animals_hetero[:,i]=np.nansum(epochs_con_this_animal_hetero,axis=1)
    ##calculate preference index: positive value means prefer L object and negative value means prefer R object
    if len(object_of_interest)==1:
        if len(trial_type_of_interest)==1:
            #single homogeneous trials
            epochs_con_all_animals=epochs_forL_all_animals_homo
            epochs_exp_all_animals=epochs_forR_all_animals_homo
            epochs_forL_all_animals=epochs_forL_all_animals_homo
            epochs_forR_all_animals=epochs_forR_all_animals_homo
        elif len(trial_type_of_interest)==2:
            #multiple homogeneous trials
            epochs_con_all_animals=epochs_forL_all_animals_homo
            epochs_exp_all_animals=epochs_forR_all_animals_homo
            epochs_forL_all_animals=epochs_forL_all_animals_homo
            epochs_forR_all_animals=epochs_forR_all_animals_homo
        else:
            Warning("unknown trial naming")
            left_right_preference_across_animals,exp_con_preference_across_animals=(np.nan,np.nan)
    elif len(object_of_interest)==2:
        if len(trial_type_of_interest)==1:
            #single heterogeneous trials
            epochs_exp_all_animals=epochs_forR_all_animals_hetero
            epochs_con_all_animals=epochs_forL_all_animals_hetero
            epochs_forL_all_animals=epochs_forL_all_animals_hetero
            epochs_forR_all_animals=epochs_forR_all_animals_hetero
        elif len(trial_type_of_interest)==2:
            #multiple heterogeneous trials
            epochs_exp_all_animals=epochs_exp_all_animals_hetero
            epochs_con_all_animals=epochs_con_all_animals_hetero
            epochs_forL_all_animals=epochs_forL_all_animals_hetero
            epochs_forR_all_animals=epochs_forR_all_animals_hetero
        else:
            Warning("unknown trial naming")
            left_right_preference_across_animals,exp_con_preference_across_animals=(np.nan,np.nan)
    else:
        Warning("unknown trial naming")
        left_right_preference_across_animals,exp_con_preference_across_animals=(np.nan,np.nan)


    if exclude_extreme_index==True:
        animal_no_extreme=(epochs_exp_all_animals==0) | (epochs_con_all_animals==0)
        epochs_exp_all_animals[animal_no_extreme]=np.nan
        epochs_con_all_animals[animal_no_extreme]=np.nan
        animal_no_extreme=(epochs_forL_all_animals==0) | (epochs_forR_all_animals==0)
        epochs_forL_all_animals[animal_no_extreme]=np.nan
        epochs_forR_all_animals[animal_no_extreme]=np.nan

    exp_con_preference_across_animals=(epochs_exp_all_animals-epochs_con_all_animals)/(epochs_exp_all_animals+epochs_con_all_animals)
    left_right_preference_across_animals=(epochs_forL_all_animals-epochs_forR_all_animals)/(epochs_forL_all_animals+epochs_forR_all_animals)
    plot_epochs_time(epochs_exp_all_animals,epochs_con_all_animals,epochs_forL_all_animals,epochs_forR_all_animals,analysis_methods,fig_name,data_color,thresholds)
    return left_right_preference_across_animals,exp_con_preference_across_animals,epochs_exp_all_animals_hetero,epochs_con_all_animals_hetero,epochs_forL_all_animals,epochs_forR_all_animals

if __name__ == "__main__":
    file_path='dataframes_list.pkl'
    with open(file_path, 'rb') as f:
        relative_pos_all_animals = pickle.load(f)
    json_file = "./analysis_methods_dictionary.json"
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    #trial_type_of_interest=['LeaderLocust']
    trial_type_of_interest=['LocustBand']
    analysis_methods.update({"frequency_based_preference_index":False})
    calculate_preference_index(relative_pos_all_animals,trial_type_of_interest,analysis_methods,thresholds=[4,5,6,7,8],this_vr='all')