import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os

save_path = "/Users/aljoscha/Downloads/locustVR_data"
df1 = pd.read_pickle(os.path.join(save_path, 'locustvr_data.pkl'))
df1 = df1.iloc[::10, :]  # Use every 10th row

for aha in range(5, 6):
    for yep in range(16, 17):
        df = df1.loc[
            (df1['animal_id'] == 5) &
            (df1['trial_id'] == yep) &
            (df1["ts"] >= 0) & (df1["ts"] <= 3000),
            ['X_aligned', 'Y_flip', 'ts', "heading", "constant_speed", "constant_distance"]
        ]

        time = df['ts'].to_numpy()
        X_focal = -df['X_aligned'].to_numpy()
        Y_focal = -df['Y_flip'].to_numpy()

        angle_0 = df['heading'].iloc[0]
        dist = df['constant_distance'].iloc[0]
        speed = df['constant_speed'].iloc[0]

        def animate_agent_cd(t, v, d, x_focal, y_focal, side):
            X = np.cos(np.radians(60)) * v * (t/100) + x_focal + np.cos(np.radians(60)) * d
            Y = np.sin(side * np.radians(-60)) * v * (t/100) + y_focal + np.sin(side * np.radians(-60)) * d
            return [X, Y]

        agent_cd = animate_agent_cd(time, 0, dist, X_focal, Y_focal, -1)
        agent_cs = animate_agent_cd(time, speed, dist, X_focal, Y_focal, 1)

        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        scat_focal, = ax.plot([], [], 'o', color='C0', label='Focal', markersize=10)
        scat_cd, = ax.plot([], [], '.', color='C1', label='Agent constant', markersize=15)
        scat_cs, = ax.plot([], [], '.', color='C2', label='Agent receding', markersize=15)
        line_segments = []  # store line artists

        # Add new trail Line2D objects
        trail_focal, = ax.plot([], [], '-', color='C0', alpha=0.6, linewidth=4)
        trail_cd, = ax.plot([], [], '-', color='C1', alpha=0.6, linewidth=3)
        trail_cs, = ax.plot([], [], '-', color='C2', alpha=0.6, linewidth=3)

        def init():
            ax.set_xlim(np.min(X_focal) - 50, np.max(X_focal) + 50)
            ax.set_ylim(np.min(Y_focal) - 50, np.max(Y_focal) + 50)
            return scat_focal, scat_cd, scat_cs, trail_focal, trail_cd, trail_cs


        def update(frame):
            # Remove old lines
            for line in line_segments:
                line.remove()
            line_segments.clear()

            # Current scatter points
            scat_focal.set_data([X_focal[frame]], [Y_focal[frame]])
            scat_cd.set_data([agent_cd[0][frame]], [agent_cd[1][frame]])
            scat_cs.set_data([agent_cs[0][frame]], [agent_cs[1][frame]])

            # Growing trails
            trail_focal.set_data(X_focal[:frame + 1], Y_focal[:frame + 1])
            trail_cd.set_data(agent_cd[0][:frame + 1], agent_cd[1][:frame + 1])
            trail_cs.set_data(agent_cs[0][:frame + 1], agent_cs[1][:frame + 1])

            # Just draw the current connection lines (no trail)
            l1, = ax.plot(
                [X_focal[frame], agent_cd[0][frame]],
                [Y_focal[frame], agent_cd[1][frame]],
                color='grey', alpha=0.5, linestyle=':', linewidth=2
            )
            l2, = ax.plot(
                [X_focal[frame], agent_cs[0][frame]],
                [Y_focal[frame], agent_cs[1][frame]],
                color='grey', alpha=0.5, linestyle=':', linewidth=2
            )
            line_segments.extend([l1, l2])

            return scat_focal, scat_cd, scat_cs, trail_focal, trail_cd, trail_cs, *line_segments

        ani = animation.FuncAnimation(
            fig, update, frames=len(time),
            init_func=init, blit=True, interval=40
        )

        # Show animation
        plt.legend(fontsize=14, loc='upper left')
        plt.tight_layout()
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.show()
        # Optional: Save animation to file
        ani.save(os.path.join(save_path, f"animation_A{aha}_T{yep}.mp4"), fps=30, dpi=300)
