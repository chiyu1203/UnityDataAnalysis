from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
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
    object_of_interest=trial_type_of_interest[0].split("_x_")
    if len(object_of_interest)==1 and len(trial_type_of_interest)==2:
        fig_name=f"preference_index_VR{this_vr}_homegenous_trials_multiple_condition_{trial_type_of_interest[0]}_and_{trial_type_of_interest[1]}"
        data_color='k'
    elif len(object_of_interest)==2 and len(trial_type_of_interest)==2:
        fig_name=f"preference_index_VR{this_vr}_heterogenous_trials_multiple_condition_top_{object_of_interest[1]}_bottom_{object_of_interest[0]}"
        data_color='b'
    elif len(object_of_interest)==1 and len(trial_type_of_interest)==1:
        fig_name=f"preference_index_VR{this_vr}_homegenous_trials_single_condition_object_type_{object_of_interest[0]}"
        data_color='k'
    elif len(object_of_interest)==2 and len(trial_type_of_interest)==1:
        fig_name=f"preference_index_VR{this_vr}_heterogenous_trials_single_condition_top_{object_of_interest[1]}_bottom_{object_of_interest[0]}"
        data_color='b'
    else:
        print("error. trial type or object type used in the analysis should be no more than 2")
    preference_plot, (ax1,ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 4), tight_layout=True
    )
    xaxis_points=np.repeat(thresholds,left_right_preference_across_animals.shape[1]).reshape(len(thresholds),left_right_preference_across_animals.shape[1])
    ax1.scatter(xaxis_points,left_right_preference_across_animals,alpha=0.2,c=data_color)
    mean_index=np.mean(xaxis_points,axis=1)
    mean_response=np.nanmean(left_right_preference_across_animals,axis=1)
    sem_response=np.nanstd(left_right_preference_across_animals, axis=1, ddof=1) / np.sqrt(left_right_preference_across_animals.shape[1])
    ax1.errorbar(
        mean_index,
        mean_response,
        yerr=sem_response,
        c=data_color,
        fmt="o",
        elinewidth=2,
        capsize=3,
    )
    ax1.set(
        ylabel="(postive means prefer left)",
        yticks=[-1,0,1],
        xticks=thresholds,
        xlabel="ROI size (cm)",
        title="spatial preference",
        xlim=(min(thresholds)-1,max(thresholds)+1)
    )
    ax2.scatter(xaxis_points,exp_con_preference_across_animals,alpha=0.2,c="r")
    mean_index=np.mean(xaxis_points,axis=1)
    mean_response=np.nanmean(exp_con_preference_across_animals,axis=1)
    sem_response=np.nanstd(exp_con_preference_across_animals, axis=1, ddof=1) / np.sqrt(exp_con_preference_across_animals.shape[1])
    ax2.errorbar(
        mean_index,
        mean_response,
        yerr=sem_response,
        c="r",
        fmt="o",
        elinewidth=2,
        capsize=3,
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
        preference_plot.savefig(f"{fig_name}.png")
        preference_plot.savefig(f"{fig_name}.svg")
    plt.show()
def plot_epochs_time(epochs_exp,epochs_con,epochs_L,epochs_R,analysis_methods,fig_name,data_color,thresholds=[4,5,6,7,8],this_vr='all'):
    ##try using seaborn to plot the data next time
    save_output=analysis_methods.get("save_output")
    camera_fps=analysis_methods.get("camera_fps")
    num_subplots=len(thresholds)
    scatterplots, axes = plt.subplots(
        nrows=1, ncols=num_subplots*2, figsize=(40, 4), tight_layout=True
    )
    #limits=[15,20,25,30,35]
    limits=[25,30,35,40,45]
    for i in range(epochs_exp.shape[0]):
        axes[i].scatter(epochs_con[i,:]/camera_fps,epochs_exp[i,:]/camera_fps,c='r')
        axes[i].set(
            xlabel="con (sec)",
            ylabel="exp (sec)",
            title=f"threshold {thresholds[i]} cm",
            xlim=(0,limits[i]),ylim=(0,limits[i]),
            xticks=[0,limits[i]//2,limits[i]],yticks=[0,limits[i]//2,limits[i]],
        )
        axes[i].plot([0, 1], [0, 1], transform=axes[i].transAxes, color='gray', linestyle='--')
        axes[i+num_subplots].scatter(epochs_L[i,:]/camera_fps,epochs_R[i,:]/camera_fps,c=data_color)
        axes[i+num_subplots].set(
            xlabel="L (sec)",
            ylabel="R (sec)",
            title=f"threshold {thresholds[i]} cm",
            xlim=(0,limits[i]),ylim=(0,limits[i]),
            xticks=[0,limits[i]//2,limits[i]],yticks=[0,limits[i]//2,limits[i]],
        )
        axes[i+num_subplots].plot([0, 1], [0, 1], transform=axes[i+num_subplots].transAxes, color='gray', linestyle='--')
    if save_output==True:
        scatterplots.savefig(f"{fig_name}.png")
        scatterplots.savefig(f"{fig_name}.svg")

def calculate_preference_index(relative_pos_all_animals,trial_type_of_interest,analysis_methods,thresholds=[4,5,6,7,8],this_vr='all'):
    spatial_preference_animals=np.zeros((len(thresholds),len(relative_pos_all_animals),relative_pos_all_animals[0]['type'].unique().shape[0]))
    spatial_preference_animals[:]=np.nan
    epochs_forL_all_animals_homo=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forR_all_animals_homo=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forL_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_forR_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_exp_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
    epochs_con_all_animals_hetero=np.zeros((len(thresholds),len(relative_pos_all_animals)))
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
        if len(trial_type_list)<4:
            print(f"animal {i} only three follow epochs from {len(trial_type_list)} trial types")
        homo_no=0
        hetero_no=0
        epochs_forL_this_animal_homo=np.zeros((len(thresholds),2))
        epochs_forR_this_animal_homo=np.zeros((len(thresholds),2))
        epochs_forL_this_animal_hetero=np.zeros((len(thresholds),2))
        epochs_forR_this_animal_hetero=np.zeros((len(thresholds),2))
        epochs_exp_this_animal_hetero=np.zeros((len(thresholds),2))
        epochs_con_this_animal_hetero=np.zeros((len(thresholds),2))
        for key,grp in relative_pos_this_animal.groupby('type'):
            #Note1: group by sort type first based on alphabetical order, and then based on length of the string
            relative_distance=LA.norm(np.vstack((grp["x"].values,grp["y"].values)),axis=0)
            epochs_forL_array=np.zeros((len(thresholds)))
            epochs_forR_array=np.zeros((len(thresholds)))
            epochs_exp_array=np.zeros((len(thresholds)))
            epochs_con_array=np.zeros((len(thresholds)))
            for j,this_threshold in enumerate(thresholds):
                ##return agent ID when relative distance is within the threshold
                ##sum agent ID to get epochs for the Right object because agent ID is 0 for the right object and 1 for the left object
                epochs_forL=np.sum(grp[relative_distance<this_threshold]["agent_id"].values)
                ##epochs for the L object comes from the rest of element in the array
                epochs_forR=grp[relative_distance<this_threshold]["agent_id"].values.shape[0]-epochs_forL
                ##assign epochs_exp or epochs_con based on the trial type, if the trial type is not of interest, assign NaN
                if len(trial_type_of_interest)==2:
                    if key==trial_type_of_interest[1]:
                        epochs_exp=epochs_forL
                        epochs_con=epochs_forR
                    elif key==trial_type_of_interest[0]:
                        epochs_exp=epochs_forR
                        epochs_con=epochs_forL
                    else:
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
        #print(epochs_forL_all_animals_homo)
    ##calculate preference index: positive value means prefer L object and negative value means prefer R object
    if len(object_of_interest)==1:
        if len(trial_type_of_interest)==1:
            #single homo
            epochs_con_all_animals=epochs_forL_all_animals_homo
            epochs_exp_all_animals=epochs_forR_all_animals_homo
            epochs_forL_all_animals=epochs_forL_all_animals_homo
            epochs_forR_all_animals=epochs_forR_all_animals_homo
        elif len(trial_type_of_interest)==2:
            #multiple homo
            epochs_con_all_animals=epochs_forL_all_animals_homo
            epochs_exp_all_animals=epochs_forR_all_animals_homo
            epochs_forL_all_animals=epochs_forL_all_animals_homo
            epochs_forR_all_animals=epochs_forR_all_animals_homo
        else:
            Warning("unknown trial naming")
            left_right_preference_across_animals,exp_con_preference_across_animals=(np.nan,np.nan)
    elif len(object_of_interest)==2:
        if len(trial_type_of_interest)==1:
            #multiple homo
            epochs_exp_all_animals=epochs_forR_all_animals_hetero
            epochs_con_all_animals=epochs_forL_all_animals_hetero
            epochs_forL_all_animals=epochs_forL_all_animals_hetero
            epochs_forR_all_animals=epochs_forR_all_animals_hetero
        elif len(trial_type_of_interest)==2:
            #multiple hetero
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

    exp_con_preference_across_animals=(epochs_exp_all_animals-epochs_con_all_animals)/(epochs_exp_all_animals+epochs_con_all_animals)
    left_right_preference_across_animals=(epochs_forL_all_animals-epochs_forR_all_animals)/(epochs_forL_all_animals+epochs_forR_all_animals)
    plot_epochs_time(epochs_exp_all_animals,epochs_con_all_animals,epochs_forL_all_animals,epochs_forR_all_animals,analysis_methods,fig_name,data_color,thresholds)
    return left_right_preference_across_animals,exp_con_preference_across_animals,epochs_exp_all_animals_hetero,epochs_con_all_animals_hetero