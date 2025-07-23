import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# time = np.linspace(0, 10, 10)

# X_focal = time
# Y_focal = np.sin(time)



save_path = "/Users/aljoscha/Downloads/locustVR_data"
df1 = pd.read_pickle(os.path.join(save_path, 'locustvr_data.pkl'))
print(df1["constant_speed"].unique())
df1 = df1.loc[df1["constant_speed"] == 1, :]
# for aha in range(1, 2):
    # for yep in range(16, 17):
#         df = df1.loc[(df1['animal_id'] == 5) & (df1['trial_id'] == yep) & (df1["ts"] >= 0) & (df1["ts"] <= 6000), ['X_aligned', 'Y_flip', 'ts', "heading", "constant_speed", "constant_distance"]]

for aha in df1['animal_id'].unique():
    print(aha)
    for yep in df1.loc[df1["animal_id"] == aha, 'trial_id'].unique():
        df = df1.loc[
            (df1['animal_id'] == aha) &
            (df1['trial_id'] == yep) &
            (df1["ts"] >= 0) &
            (df1["ts"] <= 6000),
            ['X_aligned', 'Y_flip', 'ts', "heading", "constant_speed", "constant_distance"]
        ]
        df = df.iloc[::20, :]

        time = df['ts'].to_numpy()
        X_focal = -df['X_aligned'].to_numpy()
        Y_focal = -df['Y_flip'].to_numpy()
        # angle_0 = df.loc[df['ts'] == 0, 'heading']  # Get the initial heading angle
        # print("hallo", angle_0)
        angle_0 = df['heading'].iloc[0]  # Get the initial heading angle
        dist = df['constant_distance'].iloc[0]  # Get the initial heading angle
        speed = df['constant_speed'].iloc[0]  # Get the initial heading angle

        def animate_agent_cd(t, v, d, x_focal, y_focal, side, a):
            X = np.cos(np.radians(60)) * v * (t/100) + x_focal + np.cos(np.radians(60)) * d
            Y = np.sin(side * np.radians(-60)) * v * (t/100) + y_focal + np.sin(side * np.radians(-60)) * d
            return [X, Y]

        plt.scatter(X_focal, Y_focal, c=time, cmap="viridis", s=30)
        agent_cd = animate_agent_cd(time, 0, dist, X_focal, Y_focal, -1, angle_0)
        # plt.scatter(agent_cd[0], agent_cd[1], label='Agent CD', c=time, cmap="viridis", marker='^')
        plt.scatter(agent_cd[0], agent_cd[1], label='Agent CD', color="C3", marker='.')
        agent_cs = animate_agent_cd(time, speed, dist, X_focal, Y_focal, 1, angle_0)
        # plt.scatter(agent_cs[0], agent_cs[1], label='Agent CS', c=time, cmap="viridis", marker='v')
        plt.scatter(agent_cs[0], agent_cs[1], label='Agent CS', color="C1", marker='.')

        a = np.linspace(0, 50, 100)
        x32 = np.cos(np.radians(60)) * a
        y32 = np.sin(np.radians(60)) * a
        plt.plot(x32, y32)
        # time_2 = np.arange(1, 30, 4)
        # for i in time_2:
        #     plt.plot([X_focal[i], agent_cd[0][i]], [Y_focal[i], agent_cd[1][i]], color='grey', alpha=0.7, linestyle='--', linewidth=0.8)
        #     plt.plot([X_focal[i], agent_cs[0][i]], [Y_focal[i], agent_cs[1][i]], color='grey', alpha=0.7, linestyle='--', linewidth=0.8)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(save_path, f'trajectorie_A{aha}_T{yep}_CD{dist}_CS{speed}.pdf'), format='pdf')
        plt.close()