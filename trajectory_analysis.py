#introduce customised functions
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)
# colormap_name = "viridis"

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

def plot_scatter_violin(data,sequence_config_this_condition):
    data = data[~np.isnan(data)]
    fig, (ax,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(2,4), dpi=300,sharey=True)
    ax.scatter([0] * data.shape[0],data,s=1)
    #ax.violinplot(this_data[0,:],showmedians=True,showmeans=True)
    ## customised violin plot is adapted from https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
    if 'agent' in sequence_config_this_condition:
        agent_name=sequence_config_this_condition.split('_x_')[1]
        this_ylabel=f"agent preference (+ means for {agent_name})"    
    else:
        this_ylabel="left/right preference (+ means for left)"
    ax.set(
        ylabel=this_ylabel,
        ylim=(-1.2,1.2),
        yticks=([-1, 0, 1]),
        xticks=([]),
    )
    ax.yaxis.label.set(fontsize=6)
    parts = ax2.violinplot(
            data, showmeans=False, showmedians=False,
            showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor("#000000")
        pc.set_edgecolor('black')
        pc.set_alpha(0.3)
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    means = np.mean(data, axis=0)
    whiskers=np.array([adjacent_values(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax2.scatter(1, medians, marker='o', color='white', s=30, zorder=3)
    ax2.scatter(1, means, marker='o', color='grey', s=30, zorder=3)
    ax2.vlines(1, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax2.vlines(1, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    this_sequence_config=sequence_config_this_condition.split(".")[0]
    fig_name = f"{this_sequence_config}_preference_analysis.svg"
    fig.savefig(fig_name)

def plot_sercansincos(df,analysis_methods,parameters,variable_name,vr_num='all'):
    save_output= analysis_methods.get("save_output")
    scene_name=analysis_methods.get("experiment_name")
    if analysis_methods.get("active_trials_only"):
        active_trial='active_trials'
    else:
        active_trial=''
    
    cos = df["cos"]
    sin = df["sin"]
    if 'density' in df.columns:
        density=df["density"].unique()[0]
        cos_fig_name=f"{vr_num}_cos_{scene_name}_{variable_name}_{parameters}_density_{int(density)}{active_trial}.svg"
        sin_fig_name=f"{vr_num}_sin_{scene_name}_{variable_name}_{parameters}_density_{int(density)}{active_trial}.svg" 
    elif scene_name=='choice':
        cos_fig_name=f"{vr_num}_cos_{scene_name}_{variable_name}_{parameters}_single_target_{df['type'].values[0]}{active_trial}.svg"
        sin_fig_name=f"{vr_num}_sin_{scene_name}_{variable_name}_{parameters}_single_target_{df['type'].values[0]}{active_trial}.svg"

    fig, ax = plt.subplots(dpi=300, figsize=(1.1,0.25))
    #plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['font.family'] = 'Arial'
    plt.set_cmap('cividis')
    # Set the axis line width to 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['axes.linewidth'] = 2
    plt.subplots_adjust(bottom=0.4)
    sns.kdeplot(cos, cut=0, color="#21918c", fill=True, alpha=0.9)
    plt.xlim(-1,1)
    plt.title("r cos\u03F4")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    plt.ylabel("")
    plt.xlabel("")    
    if save_output==True:
        plt.savefig(cos_fig_name)
    fig, ax = plt.subplots(dpi=300, figsize=(1.1,0.25))
    plt.subplots_adjust(bottom=0.4)
    sns.kdeplot(sin, cut=0, color="#21918c",  fill=True, alpha=0.9)#),lw=1,
    plt.xlim(1,-1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks(rotation = 90)
    ax.set_yticks([])
    plt.ylabel("")
    plt.xlabel("")
    plt.title("r sin\u03F4")
    if save_output==True:
        plt.savefig(sin_fig_name)
    plt.show()
def plot_sercantrajec(dfXY,analysis_methods,parameters,parameter_name,trajec_lim=1000,vr_num='all'):
    save_output= analysis_methods.get("save_output")
    scene_name=analysis_methods.get("experiment_name")
    if analysis_methods.get("active_trials_only"):
        active_trial='active_trials'
    else:
        active_trial=''
    
    a = dfXY.groupby('VR')
    if 'density' in dfXY.columns:
        density=dfXY["density"].unique()[0]
        print(f"The density of this trial is {density}")
        #     fig_name=f"{vr_num}_summary_trajectory_{scene_name}_{parameter_name}_{parameters}_{active_trial}.png"        
        # else:
        fig_name=f"{vr_num}_summary_trajectory_{scene_name}_{parameter_name}_{parameters}_density_{int(density)}{active_trial}.png"
    else:
        fig_name=f"{vr_num}_summary_trajectory_{scene_name}_{parameter_name}_{parameters}_single_target_{dfXY['type'].values[0]}{active_trial}.png"

    fig, ax = plt.subplots(figsize=(3,3), dpi=300) 
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['axes.linewidth'] = 2
    # Get the colormap
    cmap = plt.get_cmap('viridis')
    lw=0.5
    #plt.style.use('dark_background') 
    for i, (key2, grp2) in enumerate(a):
        color = cmap(i/ len(a))
        plt.plot(grp2["X"].values, grp2["Y"].values, color=color, linewidth=lw)

    if type(parameters)==tuple:
        this_variable=parameters[0]
    else:
        this_variable=parameters


        # Here plot agent's trajectory with hardcode parameters
    if scene_name=='choice' and dfXY['type'].values[0] !='empty_trial':
        agent_speed=2
        radial_distance=8
        duration=60
        travel_direction=this_variable*-np.pi/180#the radian circle is clockwise in Unity, so 45 degree should be used as -45 degree in the regular radian circle
        radial_distance_b=np.cos(travel_direction)*radial_distance
        delta_cos=np.cumsum(np.repeat(np.cos(travel_direction)*agent_speed, duration))
        agent_cos=radial_distance_b+delta_cos
        radial_distance_b=np.sin(travel_direction)*radial_distance
        delta_sin=np.cumsum(np.repeat(np.sin(travel_direction)*agent_speed, duration))
        agent_sin=radial_distance_b+delta_sin
        plt.plot(agent_cos, agent_sin, color='k', linewidth=lw)

    plt.xlim(-1*trajec_lim, trajec_lim)
    plt.ylim(-1*trajec_lim, trajec_lim)
    plt.yticks([-1*trajec_lim, 0, trajec_lim])
    plt.xticks([-1*trajec_lim, 0, trajec_lim])
        # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal')                                                  
    if save_output==True:    
        plt.savefig(fig_name)
    plt.show()

def plot_travel_distance_set(df_all,analysis_methods,variable_name,y_axis_lim=[0.1,1000]):
    colormap_name = "viridis"
    COL = MplColorHelper(colormap_name, 0, 10)
    
    fig, (ax1, ax2,ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=(18, 6), tight_layout=True
)
    colour_code=analysis_methods.get("graph_colour_code")
    if variable_name=='kappa':
        ax1.set_xscale('log')
    #ax1.set_yscale('log')
    ax1.set_ylim([y_axis_lim[0],y_axis_lim[1]])
    ax1.set(
        yticks=[y_axis_lim[0], y_axis_lim[1]],
        ylabel="Travel distance per second",
        xticks=sorted(df_all[0][variable_name].unique()),
        xlabel=variable_name,
    )
    ax2.set(
        ylabel="trial n",
        xlabel="interval prior to trial n",
        xlim=[y_axis_lim[0],y_axis_lim[1]],
        ylim=[y_axis_lim[0],y_axis_lim[1]],
        title="travel distance per second",
    )
    ax3.set(
        ylabel="trial n",
        xlabel="trial n-1",
        xlim=[y_axis_lim[0],y_axis_lim[1]],
        ylim=[y_axis_lim[0],y_axis_lim[1]],
        title="travel distance per second",
    )
    #ax1.gca().set_aspect('equal')
    for id in np.arange(len(df_all)):
        df=df_all[id]
        viridis_code=df['color_code']*10#normalise the code from 1 to 10
        viridis_code=viridis_code.astype('int')
        #set some thresholds to remove bad tracking 
        #df.loc[(df["distTotal"]<50.0) | (df["loss"]> 0.05), "distTotal"] = np.nan
        if df.iloc[0]["VR"].startswith('VR1'):
            vr_color=colour_code[0]
        elif df.iloc[0]["VR"].startswith('VR2'):
            vr_color=colour_code[1]
        elif df.iloc[0]["VR"].startswith('VR3'):
            vr_color=colour_code[2]
        else:
            vr_color=colour_code[3]

        #ax1.scatter(df.iloc[::2][variable_name], df.iloc[::2]["distTotal"]/df.iloc[::2]["duration"],c='k')

        ax1.scatter(df.iloc[1::2][variable_name], df.iloc[1::2]["distTotal"]/df.iloc[1::2]["duration"],c=COL.get_rgb(viridis_code[1::2]))
        if df.shape[0] % 2 == 0:
            ax2.scatter(df.iloc[::2]["distTotal"]/df.iloc[::2]["duration"],df.iloc[1::2]["distTotal"]/df.iloc[1::2]["duration"],c=COL.get_rgb(viridis_code[1::2]))
        else:
            ax2.scatter(df.iloc[:-1:2]["distTotal"]/df.iloc[:-1:2]["duration"],df.iloc[1::2]["distTotal"]/df.iloc[1::2]["duration"],c=COL.get_rgb(viridis_code[1::2]))
        ax3.scatter(df.iloc[1:-2:2]["distTotal"]/df.iloc[1:-2:2]["duration"],df.iloc[3::2]["distTotal"]/df.iloc[3::2]["duration"],c=COL.get_rgb(viridis_code[3::2]))
    plt.show()

def plot_circular_histrogram(df,analysis_methods,parameters,variable_name,vr_num='all'):
    save_output= analysis_methods.get("save_output")
    scene_name=analysis_methods.get("experiment_name")
    if analysis_methods.get("active_trials_only"):
        active_trial='active_trials'
    else:
        active_trial=''
    
    angles = df["mean_angle"]
    if 'density' in df.columns:
        density=df["density"].unique()[0]
        hist_fig_name=f"{vr_num}_circular_hist_{scene_name}_{variable_name}_{parameters}_density_{int(density)}{active_trial}.svg"
    elif scene_name=='choice':
        hist_fig_name=f"{vr_num}_circular_hist_{scene_name}_{variable_name}_{parameters}_single_target_{df['type'].values[0]}{active_trial}.svg"

    ax = plt.subplot(111, polar=True)
    ax.hist(angles, bins=24, alpha=0.75)
    #ax.set_xticks([])
    ax.set_yticks([5,10])
    ax.set_xticklabels([])
    #ax.set_title(f'banding direction: {parameters}')
    if save_output==True:
        plt.savefig(hist_fig_name)
    plt.show()
def plot_travel_histrogram(df,analysis_methods,parameters,variable_name,vr_num='all'):
    save_output= analysis_methods.get("save_output")
    scene_name=analysis_methods.get("experiment_name")
    if analysis_methods.get("active_trials_only"):
        active_trial='active_trials'
    else:
        active_trial=''
    distTotal = df["distTotal"]
    if 'density' in df.columns:
        density=df["density"].unique()[0]
        hist_fig_name=f"{vr_num}_travel_hist_{scene_name}_{variable_name}_{parameters}_density_{int(density)}{active_trial}.svg"
    elif scene_name=='choice':
        hist_fig_name=f"{vr_num}_travel_hist_{scene_name}_{variable_name}_{parameters}_single_target_{df['type'].values[0]}{active_trial}.svg"

    #fig, ax = plt.subplots(dpi=300, figsize=(1,1))
    fig, ax = plt.subplots(figsize=(5,2))
    ax.hist(distTotal, bins=24, alpha=0.75)
    ax.set_xticks([50,100])
    #ax.set_yticks([5,10])
    #ax.set_xticklabels([])
    #ax.set_title(f'banding direction: {parameters}')
    if save_output==True:
        plt.savefig(hist_fig_name)
    plt.show()

def plot_pi_oi_comparison(all_PIs,all_OIs,all_tortuosity,all_PIs_follow_only,analysis_methods,travel_distance=None,parameters='',parameter_name='mu',vr_num='all'):
    travel_distance=np.ones(all_OIs.shape[1])
    save_output= analysis_methods.get("save_output")
    scene_name=analysis_methods.get("experiment_name")
    if analysis_methods.get("active_trials_only"):
        active_trial='active_trials'
    else:
        active_trial=''
    alpha_values=np.ones(all_OIs.shape[1])
    if np.sum(travel_distance)>travel_distance.shape[0]:
        alpha_values[(travel_distance < 2500)] = 0.2
        alpha_values[(travel_distance >= 5000) & (travel_distance < 15000)] = 0.6
        #travel_distance[(travel_distance >= 10000) & (travel_distance < 15000)] = 0.8
        alpha_values[(travel_distance >= 15000)] = 1
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 14))
    ax1,ax2,ax3,ax4,ax5,ax6= axes.flatten()
    ax1.scatter(all_PIs[1],all_tortuosity[2],c='g',alpha=alpha_values,label='ISIs')
    ax1.errorbar(all_PIs[1],all_tortuosity[2],yerr=all_tortuosity[3],c='g',fmt='o')
    ax1.scatter(all_PIs[0],all_tortuosity[0],c='k',alpha=alpha_values,label='Trials')##stim
    ax1.errorbar(all_PIs[0],all_tortuosity[0],yerr=all_tortuosity[1],c='k',fmt='o')
    ax2.scatter(all_OIs[1],all_tortuosity[2],c='g',label='ISIs')
    ax2.errorbar(all_OIs[1],all_tortuosity[2],yerr=all_tortuosity[3],c='g',fmt='o')
    ax2.scatter(all_OIs[0],all_tortuosity[0],c='k',label='Trials')##stim
    ax2.errorbar(all_OIs[0],all_tortuosity[0],yerr=all_tortuosity[1],c='k',fmt='o')
    ax3.scatter(all_OIs[1],all_PIs[1],c='g',alpha=alpha_values,label='ISIs')
    ax3.scatter(all_OIs[0],all_PIs[0],c='k',alpha=alpha_values,label='Trials')##stim
    ax4.scatter(all_PIs[0],all_PIs[1],c='b',alpha=alpha_values,label='preference index')
    ax4.scatter(all_OIs[0],all_OIs[1],c='r',alpha=alpha_values,label='optomotor index')
    ax5.scatter(all_OIs[1],all_PIs_follow_only[1],c='g',alpha=alpha_values,label='ISIs')
    ax5.scatter(all_OIs[0],all_PIs_follow_only[0],c='k',alpha=alpha_values,label='Trials')##stim
    ax6.scatter(all_PIs_follow_only[0],all_PIs_follow_only[1],c='b',alpha=alpha_values,label='(follow OF vs. Track OB)')
    ax6.scatter(all_OIs[0],all_OIs[1],c='r',alpha=alpha_values,label='optomotor index')
    ax1.set(
        xlim=(-1.2, 1.2),
        xticks=([-1, 0, 1]),
        xlabel='preference index',
        ylabel='tortuosity',
    )
    ax2.set(
        xlim=(-1.2, 1.2),
        xticks=([-1, 0, 1]),
        xlabel='optomotor index',
        ylabel='tortuosity',
    )
    ax3.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        yticks=([-1, 0, 1]),
        xticks=([-1, 0, 1]),
        xlabel='optomotor index',
        ylabel='preference index',
        aspect=('equal'),
    )
    ax4.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        yticks=([-1, 0, 1]),
        xticks=([-1, 0, 1]),
        aspect=('equal'),
        xlabel='index during trials',
        ylabel='index during ISIs',
    )
    ax5.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        yticks=([-1, 0, 1]),
        xticks=([-1, 0, 1]),
        xlabel='optomotor index',
        ylabel='preference index (follow OF vs. Track OB)',
        aspect=('equal'),
    )
    ax6.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        yticks=([-1, 0, 1]),
        xticks=([-1, 0, 1]),
        aspect=('equal'),
        xlabel='index during trials',
        ylabel='index during ISIs',
    )
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper left')
    ax5.legend(loc='upper left')
    ax6.legend(loc='upper left')
    fig.suptitle('+ pi means prefer optic flow, - means prefer target,; + oi means follow optic flow, - means go against optic flow')
    fig_name=f"{vr_num}_oi_pi_comparision_{scene_name}_{parameter_name}_{parameters}{active_trial}.svg"
    if save_output:
        fig.savefig(fig_name)