"""
locustVR_data_preperation
"""

def access_utilities(utilities_name='utilities'):
    from pathlib import Path
    import sys
    cwd = Path.cwd()
    parent_dir = cwd.resolve().parents[1]
    # print(parent_dir)
    path_utilities = Path(parent_dir) / utilities_name
    sys.path.insert(0, str(path_utilities))
    print(path_utilities)

#################################################################################################

def reindex_t_by_state_transition(df, transition_from=0, transition_to=(1, 2), index_col='ts', state_col='state_type'):
    import numpy as np
    df = df.copy()
    state = df[state_col].to_numpy()

    # Step 1: Detect valid transitions from 0 → 1 or 2
    prev = np.roll(state, 1)
    prev[0] = transition_from
    is_transition = (prev == transition_from) & np.isin(state, transition_to)

    # Step 2: Mark all positions as belonging to a segment
    group_id = np.zeros_like(state, dtype=int)
    group_id[:] = -1  # init with -1 (unassigned)
    current_group = 0

    t = np.full_like(state, np.nan, dtype=float)

    for idx in np.flatnonzero(is_transition):
        # Walk backward while state == transition_from
        i = idx - 1
        while i >= 0 and state[i] == transition_from and group_id[i] == -1:
            group_id[i] = current_group
            i -= 1

        # Assign transition point and forward as long as state remains in transition_to
        i = idx
        while i < len(state) and state[i] in transition_to and group_id[i] == -1:
            group_id[i] = current_group
            i += 1

        current_group += 1

    # Step 3: Assign relative t values within each group
    df['group'] = group_id
    for g in range(current_group):
        segment = df[df['group'] == g]
        idxs = segment.index.to_numpy()

        # Find first transition point (t=0)
        transition_idx = idxs[np.isin(state[idxs], transition_to)][0]
        df.loc[idxs, index_col] = idxs - transition_idx

    df.drop(columns='group', inplace=True)
    df[index_col] = df[index_col].astype('Int64')
    return df

###################################################################################################

def align_and_flip_heading(df, heading_col='heading', trial_col='trial_id', t_col='ts', state_col='state_type'):
    """
    Aligns and flips the heading of a DataFrame based on the first frame of each trial.
    Args:
        df
        heading_col
        trial_col
        t_col
        state_col
    Process:
        df['heading_rel']
        df['heading_rel_flip']
    Returns:
        df
    """
    import numpy as np
    df = df.copy()

    # 1. Normalize heading relative to the first frame (t == 0) of each trial
    trial_starts = df[df[t_col] == 0].set_index(trial_col)[heading_col]
    df['heading_rel'] = df[heading_col] - df[trial_col].map(trial_starts)

    # 2. Wrap to [-π, π)
    df['heading_rel'] = (df['heading_rel'] + np.pi) % (2 * np.pi) - np.pi

    # 3. Create flipped version (only flipped for state_type == 2)
    df['heading_rel_flip'] = df['heading_rel']
    flip_mask = df[state_col] == 2
    df.loc[flip_mask, 'heading_rel_flip'] = -df.loc[flip_mask, 'heading_rel_flip']

    return df

#############################################################################################

def convert_trial_label(df):
    # Extract values using regex groups
    extracted = df['trial_label'].str.extract(r'CD(\d+(?:\.\d+)?)_CS(\d+(?:\.\d+)?)')

    # Assign to new columns, converting to float or int if needed
    df['constant_distance'] = extracted[0].astype(float).astype(int)
    df['constant_speed'] = extracted[1].astype(float).astype(int)

    # Drop the original column
    df.drop(columns=['trial_label'], inplace=True)

    return df

#############################################################################################

def align_trajectories(df, trial_col='trial_id', t_col='ts', heading_col='heading', x_col='X', y_col='Y'):
    import numpy as np
    df = df.copy()

    # Ensure we get a single row per trial (first ts == 0 per trial)
    starts = (
        df[df[t_col] == 0]
        .groupby(trial_col, as_index=False)
        .first()
        .set_index(trial_col)
    )

    # Now indexing is safe
    ref_x = starts[x_col]
    ref_y = starts[y_col]
    ref_heading = starts[heading_col]

    # Map to full DataFrame
    x0 = df[trial_col].map(ref_x)
    y0 = df[trial_col].map(ref_y)
    theta0 = df[trial_col].map(ref_heading)

    # Shift and rotate
    dx = df[x_col] - x0
    dy = df[y_col] - y0

    cos_theta = np.cos(-theta0)
    sin_theta = np.sin(-theta0)

    df['X_aligned'] = cos_theta * dx - sin_theta * dy
    df['Y_aligned'] = sin_theta * dx + cos_theta * dy

    return df

#################################################################################################

def flip_symmetric_states(df, state_col='state_type', x_col='X_aligned', y_col='Y_aligned', heading_col='heading_rel'):
    import numpy as np
    df = df.copy()

    df["X_flip"] = df[x_col]
    df["Y_flip"] = df[y_col]

    # Only flip state_type == 2
    symmetric_mask = df[state_col] == 2

    grouped = df.groupby(['animal_id', 'trial_id'])

    for (animal_id, trial_id), group in grouped:
        idx_mask = (df['animal_id'] == animal_id) & (df['trial_id'] == trial_id) & symmetric_mask

        if group.empty or not idx_mask.any():
            continue

        # Get initial heading at ts == 0 (or closest available)
        trial_start = group.loc[group['ts'] == group['ts'].min()]
        if trial_start.empty:
            continue

        heading0 = trial_start[heading_col].values[0]  # in radians

        # Mirror angle: flip across heading0 + π/2
        # So the flip angle = 2 * (heading0 + π/2)
        theta = 2 * (heading0 + np.pi / 2)

        # Apply rotation
        x = df.loc[idx_mask, x_col].values
        y = df.loc[idx_mask, y_col].values

        # Rotate by -theta to mirror
        x_flip = np.cos(-theta) * x - np.sin(-theta) * y
        y_flip = np.sin(-theta) * x + np.cos(-theta) * y

        df.loc[idx_mask, "X_flip"] = x_flip
        df.loc[idx_mask, "Y_flip"] = y_flip

    return df

##################################################################################################

def compute_directness_and_direction(df, trial_col='trial_id', t_col='ts',
                                     x_col='X_aligned', y_col='Y_aligned'):
    import numpy as np
    df = df.copy()

    directness_list = []
    angle_list = []

    # Group by trial
    for _, group in df.groupby(trial_col):
        group_sorted = group.sort_values(t_col)
        x = group_sorted[x_col].to_numpy()
        y = group_sorted[y_col].to_numpy()

        if len(x) < 2:
            directness_list.extend([np.nan] * len(group))
            angle_list.extend([np.nan] * len(group))
            continue

        dx = x[-1] - x[0]
        dy = y[-1] - y[0]

        # Direct distance
        d_direct = np.hypot(dx, dy)

        # Trajectory length
        d_steps = np.hypot(np.diff(x), np.diff(y))
        d_trajectory = np.sum(d_steps)

        # Avoid division by zero
        directness = d_direct / d_trajectory if d_trajectory > 0 else np.nan
        # angle_direct = np.arctan2(dy, dx)

        # Fill values for each row in group
        n = len(group)
        directness_list.extend([directness] * n)
        # angle_list.extend([angle_direct] * n)
    df["directness"] = directness_list
    # df["angle_direct"] = angle_list

    return df

###################################################################################################

"""
heading_angle_visualisation
"""

def default_style(x_label, y_label, limits=None):
    import matplotlib.pyplot as plt
    plt.xlim(limits[0][0], limits[0][1])
    plt.ylim(limits[1][0], limits[1][1])
    plt.grid(False)
    plt.box(True)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

def generate_plot_type(plot_type):
    if plot_type == "hist":
        plt_hist = True
        plt_kde = False
    elif plot_type == "kde":
        plt_kde = True
        plt_hist = False
    else:
        raise ValueError("plot_type must be 'hist' or 'kde'")
    return plt_hist, plt_kde

def generate_time_windows(critical_time):
    time_windows = {}
    for i in range(len(critical_time)-1):
        row = {f"t1 ({critical_time[i]:.1f} to {critical_time[i+1]:.1f})": (critical_time[i], critical_time[i+1]),}
        time_windows.update(row)
    return time_windows


def plt_density(df, angle_version, time_windows, angle_bins, plt_hist=False, plt_kde=False, label2=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    for label1, (t_start, t_end) in time_windows.items():
        mask = (df['ts'] >= t_start) & (df['ts'] < t_end)
        angles = df.loc[mask, angle_version].dropna()

        if len(time_windows) > 1:
            label_plt = label1
        else:
            label_plt = label2
        if plt_hist:
            plt.hist(np.degrees(angles), bins=np.degrees(angle_bins), density=True, alpha=0.4, label=label_plt)

        if len(angles) < 10:
            continue  # Skip very small samples to avoid noisy KDEs

        # Fit KDE
        angles_deg = np.degrees(angles)
        kde = gaussian_kde(angles_deg, bw_method='scott')
        x_vals = np.linspace(-180, 180, 500)
        kde_values = kde(x_vals)
        if plt_kde:
            plt.plot(x_vals, kde_values, label=label_plt)

    plt.axvline(60, color='b', linestyle='--')
    plt.axvline(-60, color='b', linestyle='--')

    plt.xlabel('Relative heading (°)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
