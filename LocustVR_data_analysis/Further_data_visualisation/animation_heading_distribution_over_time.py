
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from tqdm import tqdm

save_path = "/Users/aljoscha/Downloads/locustVR_data"
df = pd.read_pickle(os.path.join(save_path, 'locustvr_data.pkl'))
df = df.iloc[::10, :]  # Use every 10th row

def animate_angle_density(df, angle_version="heading_rel", ts_start=-2000, ts_end=9494, angle_bins=36, window_size=100, step_size=50, fps=100):
    df = df.copy()
    df = df.sort_values('ts')  # Ensure proper time ordering
    df = df[(df['ts'] >= ts_start) & (df['ts'] <= ts_end)]
    df['angle_deg'] = np.degrees(df[angle_version])

    time_vals = np.arange(ts_start + window_size / 2, ts_end - window_size / 2, step_size)
    x_vals = np.linspace(-180, 180, 500)
    kde_results = []

    # Rolling KDE for each window
    peak_angles = []  # Store angle of max density

    # Rolling KDE for each window
    for center_ts in tqdm(time_vals, desc="Computing rolling KDE"):
        window_mask = (df['ts'] >= center_ts - window_size / 2) & (df['ts'] < center_ts + window_size / 2)
        window_data = df.loc[window_mask, 'angle_deg'].dropna()

        if len(window_data) < 10:
            kde_results.append(None)
            peak_angles.append(np.nan)
        else:
            kde = gaussian_kde(window_data, bw_method='scott')
            density = kde(x_vals)
            kde_results.append(density)
            peak_angle = x_vals[np.argmax(density)]
            peak_angles.append(peak_angle)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2)
    vline1 = ax.axvline(60, color='b', linestyle='--')
    vline2 = ax.axvline(-60, color='b', linestyle='--')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 0.02)  # Adjust depending on your data
    ax.set_xlabel('Relative heading (°)')
    ax.set_ylabel('Density')
    ax.set_title('Density of Relative Heading Over Time')

    def update(frame_idx):
        kde_vals = kde_results[frame_idx]
        if kde_vals is not None:
            line.set_data(x_vals, kde_vals)
        else:
            line.set_data([], [])
        time_text.set_text(f'time: {time_vals[frame_idx]:.0f}')
        return line, time_text

    ani = FuncAnimation(fig, update, frames=len(time_vals), interval=1000 / fps, blit=True)
    plt.tight_layout()
    # ani.save(os.path.join(save_path, 'heading_density.mp4'), fps=fps, dpi=100, writer='ffmpeg')
    plt.show()
    print(time_vals)
    print(peak_angles)
    # Plot peak angle over time
    plt.figure(figsize=(10, 4))
    plt.plot(time_vals /100, peak_angles, marker='o', linestyle='-', markersize=2)
    plt.xlabel('Time')
    plt.ylabel('Peak Angle (°)')
    # plt.title('Peak Relative Heading Over Time')
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'peak_heading_over_time.png'))
    plt.close()

print(1)


animate_angle_density(
    df=df,
    angle_version='heading_rel_flip',  # or whatever your angle column is
    ts_start=-2000,
    ts_end=9000,
    angle_bins=np.linspace(-np.pi, np.pi, 36),
    window_size=100,
    step_size=20,
    fps=40
)
