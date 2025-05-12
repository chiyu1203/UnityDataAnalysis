import sys
from pathlib import Path
current_working_directory = Path.cwd()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import select_animals_gpt,find_file,column_name_list,get_fill_between_range

def fix_data_type(all_trials):
    all_trials['id'] = all_trials['id'].astype(int)
    all_trials['mu'] = all_trials['mu'].astype(int)
    all_trials['velocity'] = all_trials['velocity'].astype(float)
    all_trials['omega'] = all_trials['omega'].astype(float)
    all_trials['normalised_v'] = all_trials['normalised_v'].astype(float)
    all_trials['normalised_omega'] = all_trials['normalised_omega'].astype(float)
    if 'density' in all_trials.columns:
        all_trials['density'] = all_trials['density'].astype(float)
    if 'X' in all_trials.columns:
        all_trials['X'] = all_trials['X'].astype(float)
    if 'Y' in all_trials.columns:
        all_trials['Y'] = all_trials['Y'].astype(float)
    return all_trials
def check_baseline_distribution(all_trials,analysis_methods,metrics_name='velocity',duration_for_baseline=3):
    monitor_fps=analysis_methods.get("monitor_fps")
    align_with_isi_onset=analysis_methods.get("align_with_isi_onset",False)
    these_baselines=[]
    for keys, this_data in all_trials.groupby(['animal_id','id']):
        #print(this_data['object'][1])
        if align_with_isi_onset:
            if this_data['object'][1]!='empty_trial':
            #if keys[1]%2==0:#for the Swarm scene to only use stim trials to get the baseline
                this_metrics=this_data[metrics_name].values
                #print(this_metrics)
                these_baselines.append(np.mean(this_metrics[-duration_for_baseline*monitor_fps:]))
        
        else:
            if this_data['object'][1]=='empty_trial':
            #if keys[1]%2!=0:#for the Swarm scene to only use ISI trials
                this_metrics=this_data[metrics_name].values
                these_baselines.append(np.mean(this_metrics[-duration_for_baseline*monitor_fps:]))
    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(9,9), tight_layout=True
    )
    axes.hist(np.vstack(these_baselines),bins=500)
    if metrics_name=='velocity':
        axes.set(xlim=(0,10),ylim=(0, 250))
    else:
        #axes.set(xlim=(-0.001,0.001),ylim=(0, 60))#if used rad
        axes.set(xlim=(-3,3),ylim=(0, 400))

    plt.minorticks_on()
    plt.show()
def split_trials(analysis_methods,all_trials,metrics_name='velocity',metrics_name2='normalised_v',walk_threshold=1,duration_for_baseline=3):
    monitor_fps=analysis_methods.get("monitor_fps")
    movement_trial_boolean=[]
    these_metrics=[]
    these_normalised_metrics=[]
    for keys, this_data in all_trials.groupby(['animal_id','id']):
        this_metrics=this_data[metrics_name].values
        baseline_metrics=np.mean(this_metrics[-duration_for_baseline*monitor_fps:])
        movement_trial_boolean.append(abs(baseline_metrics)>walk_threshold)
        these_metrics.append(this_metrics)
        these_normalised_metrics.append(this_data[metrics_name2].values)
    return movement_trial_boolean,these_metrics,these_normalised_metrics
def extract_trial_index(movement_trial_boolean,num_animals,analysis_methods):
    align_with_isi_onset=analysis_methods.get("align_with_isi_onset",False)
#int((trial_id)/2) means the number of stimulus trial
    trial_id=len(movement_trial_boolean)/num_animals
    after_movement_ith_trial=[]
    after_no_movement_ith_trial=[]
    if align_with_isi_onset:
        after_movement_ith_trial=[i+1 for i, x in enumerate(movement_trial_boolean[1::2]) if x and i % int((trial_id)/2) != (trial_id)/2-1]
        after_no_movement_ith_trial=[i+1 for i, x in enumerate(movement_trial_boolean[1::2]) if x==False and i % int((trial_id)/2) != (trial_id)/2-1]
    else:
        after_movement_ith_trial=[i for i, x in enumerate(movement_trial_boolean[::2]) if x]
        after_no_movement_ith_trial=[i for i, x in enumerate(movement_trial_boolean[::2]) if x==False]
    return after_movement_ith_trial,after_no_movement_ith_trial

def plot_visual_evoked_behaviour(these_metrics,these_normalised_metrics,after_movement_ith_trial,after_no_movement_ith_trial,analysis_methods,metrics_name='velocity',row_of_interest=None,type_key="",variable_values=None):
    exp_name=analysis_methods.get('experiment_name')
    number_frame_scene_changing=analysis_methods.get("largest_unfilled_gap",12)
    number_frame_scene_changing=10#set an arbitrary value to escape from the effect of missing value when plotting the histogram
    align_with_isi_onset=analysis_methods.get("align_with_isi_onset",False)
    save_output=analysis_methods.get("save_output",False)
    analysis_window=analysis_methods.get("analysis_window")
    monitor_fps=analysis_methods.get("monitor_fps")
    camera_fps=analysis_methods.get("camera_fps")
    all_animals=False
    tmp=np.vstack(these_metrics)
    tmp3=np.vstack(these_normalised_metrics)
    if align_with_isi_onset:
       stim_evoked_metrics=tmp[::2]
       stim_evoked_norm_metrics=tmp3[::2]
    else:
       stim_evoked_metrics=tmp[1::2]
       stim_evoked_norm_metrics=tmp3[1::2]

    if all_animals==False and type(row_of_interest)==pd.Series:
        animal_interest_stationary=[]
        animal_interest_move=[]
        for i in np.where(row_of_interest)[0].tolist():
            if i in after_no_movement_ith_trial:
                animal_interest_stationary.append(i)
            else:
                animal_interest_move.append(i)
        p3=stim_evoked_metrics[animal_interest_stationary,:]
        p4=stim_evoked_metrics[animal_interest_move,:]
        p5=stim_evoked_norm_metrics[animal_interest_stationary,:]
        p6=stim_evoked_norm_metrics[animal_interest_move,:]

    else:
        p3=stim_evoked_metrics[after_no_movement_ith_trial,:]
        p4=stim_evoked_metrics[after_movement_ith_trial,:]
        p5=stim_evoked_norm_metrics[after_no_movement_ith_trial,:]
        p6=stim_evoked_norm_metrics[after_movement_ith_trial,:]

    # plt.rcParams.update(plt.rcParamsDefault)
    # plt.rcParams.update({'font.size': 8})
    # # Set the axis line width to 2
    # plt.rcParams['ytick.major.width'] = 1
    # plt.rcParams['xtick.major.width'] = 1
    # plt.rcParams['axes.linewidth'] = 1
    # cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(18,20), tight_layout=True
    )
    ax1, ax2, ax3, ax4,ax5,ax6,ax7,ax8 = axes.flatten()
    if metrics_name=='xy':

        agent_speed=2
        agent_distance=8
        agent1_loc=(agent_distance*np.cos((np.pi/4)),agent_distance*np.sin((np.pi/4)))
        agent2_loc=(agent_distance*np.cos((-np.pi/4)),agent_distance*np.sin((-np.pi/4)))
        ax1.plot([agent1_loc[0],agent1_loc[0]+agent_speed*analysis_window[1]],[agent1_loc[1],agent1_loc[1]],'k--')
        ax1.plot([agent2_loc[0],agent2_loc[0]+agent_speed*analysis_window[1]],[agent2_loc[1],agent2_loc[1]],'k--')
        ax2.plot([agent1_loc[0],agent1_loc[0]+agent_speed*analysis_window[1]],[agent1_loc[1],agent1_loc[1]],'k--')
        ax2.plot([agent2_loc[0],agent2_loc[0]+agent_speed*analysis_window[1]],[agent2_loc[1],agent2_loc[1]],'k--')
        ax3.plot([agent1_loc[0],agent1_loc[0]+agent_speed*analysis_window[1]],[agent1_loc[1],agent1_loc[1]],'k--')
        ax3.plot([agent2_loc[0],agent2_loc[0]+agent_speed*analysis_window[1]],[agent2_loc[1],agent2_loc[1]],'k--')
        ax4.plot([agent1_loc[0],agent1_loc[0]+agent_speed*analysis_window[1]],[agent1_loc[1],agent1_loc[1]],'k--')
        ax4.plot([agent2_loc[0],agent2_loc[0]+agent_speed*analysis_window[1]],[agent2_loc[1],agent2_loc[1]],'k--')
        for i in range(p3.shape[0]):
            ax1.plot(p3[i,:],p5[i,:])
            ax3.plot(p3[i,:],p5[i,:])
            ax5.scatter(p3[i,:],p5[i,:],c=np.arange(p3.shape[1]),marker=".")
            ax7.scatter(p3[i,:],p5[i,:],c=np.arange(p3.shape[1]),marker=".")       
        for i in range(p4.shape[0]):
            ax2.plot(p4[i,:],p6[i,:])
            ax4.plot(p4[i,:],p6[i,:])
            ax6.scatter(p4[i,:],p6[i,:],c=np.arange(p4.shape[1]),marker=".")
            ax8.scatter(p4[i,:],p6[i,:],c=np.arange(p4.shape[1]),marker=".")
        ax3.set_ylim([-10,10])
        ax4.set_ylim([-10,10])
        ax3.set_xlim([-5,30])
        ax4.set_xlim([-5,30])
        ax5.set_ylim([-15,15])
        ax6.set_ylim([-15,15])
        ax5.set_xlim([-5,120])
        ax6.set_xlim([-5,120])
        ax7.set_ylim([-10,10])
        ax8.set_ylim([-10,10])
        ax7.set_xlim([-5,30])
        ax8.set_xlim([-5,30])
        ax1.set_ylim([-15,15])
        ax2.set_ylim([-15,15])
        ax1.set_xlim([-5,120])
        ax2.set_xlim([-5,120])

        #plt.hist2d(p4,p6),bins=50)
    else:
        x=np.arange(0,p3.shape[1])
        p1=np.nancumsum(p3, axis=1)/camera_fps
        p2=np.nancumsum(p4, axis=1)/camera_fps
        ax1.plot(np.transpose(p1),linewidth=0.1)
        mean_p1=np.nanmean(p1,axis=0)
        ax1.plot(mean_p1,'k',linewidth=1)
        if metrics_name=='velocity':
            circular_statistics=False
            confidence_interval=True
        else:
            circular_statistics=True
            confidence_interval=False
        dif_y1,dif_y2=get_fill_between_range(p1,confidence_interval,circular_statistics)
        ax1.fill_between(x,dif_y1,dif_y2, alpha=0.4,color='k')
        ax2.plot(np.transpose(p2),linewidth=0.1)
        mean_p2=np.nanmean(p2,axis=0)
        ax2.plot(mean_p2,'k',linewidth=1)
        dif_y1,dif_y2=get_fill_between_range(p2,confidence_interval,circular_statistics)
        ax2.fill_between(x,dif_y1,dif_y2, alpha=0.4,color='k')
        ax3.plot(np.transpose(p3),linewidth=0.1)
        mean_p3=np.nanmean(p3,axis=0)
        ax3.plot(mean_p3,'k',linewidth=1)
        dif_y1,dif_y2=get_fill_between_range(p3)
        ax3.fill_between(x,dif_y1,dif_y2, alpha=0.4,color='k')
        mean_p4=np.nanmean(p4,axis=0)
        dif_y1,dif_y2=get_fill_between_range(p4)
        ax4.plot(np.transpose(p4),linewidth=0.1)
        ax4.plot(mean_p4,'k',linewidth=1)
        ax4.fill_between(x,dif_y1,dif_y2, alpha=0.4,color='k')
        mean_p5=np.nanmean(p5,axis=0)
        dif_y1,dif_y2=get_fill_between_range(p5)
        ax5.plot(np.transpose(p5),linewidth=0.1)
        ax5.plot(mean_p5,'k',linewidth=1)
        ax5.fill_between(x,dif_y1,dif_y2, alpha=0.4,color='k')
        mean_p6=np.nanmean(p6,axis=0)
        dif_y1,dif_y2=get_fill_between_range(p6)
        ax6.plot(np.transpose(p6),linewidth=0.1)
        ax6.plot(mean_p6,'k',linewidth=1)
        ax6.fill_between(x,dif_y1,dif_y2,alpha=0.4,color='k')
        if metrics_name=='velocity' and all_animals==False:
            ylimit=10
            ylimit_log=100
            ax1.set_ylim([0,5*ylimit])
            ax2.set_ylim([0,5*ylimit])
            ax3.set_ylim([0,1*ylimit])
            ax4.set_ylim([0,1*ylimit])
        elif all_animals==False:
            ylimit=20
            ylimit_log=1000
            ax1.set_ylim([-1*ylimit,1*ylimit])
            ax2.set_ylim([-1*ylimit,1*ylimit])
            ax3.set_ylim([-1*ylimit,1*ylimit])
            ax4.set_ylim([-1*ylimit,1*ylimit])
        elif metrics_name=='velocity':
            ylimit=10
            ylimit_log=100
            ax1.set_ylim([0,5*ylimit])
            ax2.set_ylim([0,5*ylimit])
            ax3.set_ylim([0,1*ylimit])
            ax4.set_ylim([0,1*ylimit])
        else:
            ylimit=40
            ylimit_log=1000
            ax1.set_ylim([-1*ylimit,1*ylimit])
            ax2.set_ylim([-1*ylimit,1*ylimit])
            ax3.set_ylim([-1*ylimit,1*ylimit])
            ax4.set_ylim([-1*ylimit,1*ylimit])
        ax1.set(
            ylabel=f"sum of {metrics_name}",
            xlabel="Time (s)",
            xticks=[0,analysis_window[1]*monitor_fps],
            xticklabels=(['0', str(analysis_window[1])]),
        )
        ax2.set(
            ylabel=f"sum of {metrics_name}",
            xlabel="Time (s)",
            xticks=[0,analysis_window[1]*monitor_fps],
            xticklabels=(['0', str(analysis_window[1])]),
        )
        ax3.set(
            ylabel=metrics_name,
            xlabel="Time (s)",
            xticks=[0,analysis_window[1]*monitor_fps],
            xticklabels=(['0', str(analysis_window[1])]),
        )
        ax4.set(
            ylabel=metrics_name,
            xlabel="Time (s)",
            xticks=[0,analysis_window[1]*monitor_fps],
            xticklabels=(['0', str(analysis_window[1])]),
        )
        ax5.set(
            ylabel="normalised values (ratio)",
            xlabel="frame",
            ylim=[1/ylimit_log,ylimit_log],
        )
        ax6.set(
            ylabel="normalised values (ratio)",
            xlabel="frame",
            ylim=[1/ylimit_log,ylimit_log],
        )
        peak_distribution_stationary=np.argmax(abs(p3[:,number_frame_scene_changing:]), axis=1)
        peak_distribution_stationary=peak_distribution_stationary+number_frame_scene_changing

        peak_distribution_move=np.argmax(abs(p4[:,number_frame_scene_changing:]), axis=1)
        peak_distribution_move=peak_distribution_move+number_frame_scene_changing
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax7.hist(abs(peak_distribution_stationary),bins=10)
        ax7.set(
            ylabel="Count",
            xlabel="Time (s)",
            xticks=[0,analysis_window[1]*monitor_fps],
            xticklabels=(['0', str(analysis_window[1])]),
        )
        ax8.hist(abs(peak_distribution_move),bins=10)
        ax8.set(
            ylabel="Count",
            xlabel="Time (s)",
            xticks=[0,analysis_window[1]*monitor_fps],
            xticklabels=(['0', str(analysis_window[1])]),
        )
        if save_output==True:
            fig_name=f"ts_plot_{exp_name}_{variable_values}_{metrics_name}_{type_key}.png"
            fig.savefig(fig_name)
    plt.show()