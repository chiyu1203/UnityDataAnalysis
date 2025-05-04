from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#1D histogram to show the frequency of following response

def plot_follow_response_distribution(all_evaluation):
    fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(9,10), tight_layout=True
        )
    ax1, ax2, ax3, ax4 = axes.flatten()
    follow_time_tbt=all_evaluation['num_follow_epochs']/all_evaluation['number_frames']
    follow_time_aba=all_evaluation.groupby(['animal_id'])['num_follow_epochs'].sum()/all_evaluation.groupby(['animal_id'])['number_frames'].sum()
    simulated_time_aba=all_evaluation.groupby(['animal_id'])['num_chance_epochs'].sum()/all_evaluation.groupby(['animal_id'])['number_frames'].sum()
    simulated_time_tbt=all_evaluation['num_chance_epochs']/all_evaluation['number_frames']
    print("first 1/3 best of followers:", np.quantile(follow_time_aba, 0.66))
    print("middle 1/3 best of followers:", np.quantile(follow_time_aba, 0.33))
    print("first 1/3 best of follow epochs:", np.quantile(follow_time_tbt, 0.66))
    print("middle 1/3 best of follow epochs:", np.quantile(follow_time_tbt, 0.33))
    print("first 1/3 best of simulated followers:", np.quantile(simulated_time_aba, 0.66))
    print("first 1/3 best of simulated epochs:", np.quantile(simulated_time_tbt, 0.66))
    fair_follower_threshold=np.quantile(follow_time_aba, 0.33)
    good_follower_threshold=np.round(np.quantile(follow_time_aba, 0.66),5)
    bins_aba=np.linspace(0,0.5,21)
    ax1.hist(follow_time_aba,color='red',bins=bins_aba)
    ax1.axvline(x=good_follower_threshold,color="k",linestyle="--")
    if simulated_time_aba.eq(0.0).all()==False:
        ax1.hist(simulated_time_aba,color='tab:gray',alpha=0.5,bins=bins_aba)
    ax1.set(xticks=[0,0.25,0.5,0.75,1],xticklabels=(['0', '25', '50', '75', '100']),xlim=(0,0.5),yticks=[0,20],ylim=(0,20),title='proportion of time following aba')
    ax2.hist(follow_time_tbt,color='red',bins=bins_aba)
    if simulated_time_tbt.isnull().all()==False:
        ax2.hist(simulated_time_tbt,color='tab:gray',alpha=0.5,bins=bins_aba)
    ax2.set(xticks=[0,0.25,0.5,0.75,1],xticklabels=(['0', '25', '50', '75', '100']),xlim=(0,1),ylim=(0,200),title='proportion of time following tbt')
    follower_of_interest=all_evaluation.groupby(['animal_id'])['num_follow_epochs'].sum()/all_evaluation.groupby(['animal_id'])['number_frames'].sum()<good_follower_threshold
    rows_of_follower=follower_of_interest.repeat(int(all_evaluation.shape[0]/follower_of_interest.shape[0]))
    ax3.hist(all_evaluation[rows_of_follower.values]['num_follow_epochs']/all_evaluation[rows_of_follower.values]['number_frames'],density=True)
    ax3.set(xticks=[0,0.25,0.5,0.75,1],xlim=(0,1),title=f'proportion of time from animals below {good_follower_threshold}')
    follower_of_interest=all_evaluation.groupby(['animal_id'])['num_follow_epochs'].sum()/all_evaluation.groupby(['animal_id'])['number_frames'].sum()>good_follower_threshold
    rows_of_follower=follower_of_interest.repeat(int(all_evaluation.shape[0]/follower_of_interest.shape[0]))
    ax4.hist(all_evaluation[rows_of_follower.values]['num_follow_epochs']/all_evaluation[rows_of_follower.values]['number_frames'],density=True)
    ax4.set(xticks=[0,0.25,0.5,0.75,1],xlim=(0,1),title=f'proportion of time from animals above {good_follower_threshold}')
    plt.show()
    return follow_time_aba,follow_time_tbt