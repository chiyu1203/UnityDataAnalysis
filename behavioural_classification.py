import time, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(0, str(parent_dir) + "\\utilities")
from useful_tools import find_file


def classify_heading_direction(data, mu=0):
    """
    Classifies an array of angles in radians into two groups without using explicit loops.

    Parameters:
    - data: Array of angles in radians

    Returns:
    - Array of classifications: 'Group 1', 'Group 2', or 'Unclassified'
    """
    # Normalize angles to the range [-π, π]
    # ax = plt.subplot(111, polar=True)
    # ax.hist(data, bins=24, alpha=0.75, color="k")
    rotated_data = (data - np.deg2rad(-1 * mu) + np.pi) % (2 * np.pi) - np.pi
    # ax.hist(rotated_data, bins=24, alpha=0.75, color="r")
    # ax.set(title=f" vr locust heading direction: -1*{mu} degree")
    # plt.show()
    # Initialize an array of "Unclassified" labels
    labels = np.full(data.shape, "target_ob", dtype=object)

    # Create boolean masks for each group
    for_of = (-np.pi / 4 <= rotated_data) & (rotated_data < np.pi / 4)
    against_of = ((3 * np.pi / 4 <= rotated_data) & (rotated_data < np.pi)) | (
        (-3 * np.pi / 4 > rotated_data) & (rotated_data >= -np.pi)
    )
    for_left=(3 * np.pi / 4 > rotated_data) & (rotated_data >= np.pi / 4)
    for_right=(-np.pi / 4 > rotated_data) & (-3 * np.pi / 4 <= rotated_data)
    # group_3_mask = ((np.pi/4 <= data) & (data < 3*np.pi/4))
    # group_4_mask = ((-np.pi/4 > data) & (data >= -3*np.pi/4))
    # Apply classifications based on masks
    labels[for_of] = "for_of"
    labels[against_of] = "against_of"
    labels[for_left]="for_left"
    labels[for_right]="for_right"
    labels[0] = "initial_heading"
    num_for_of = sum(labels == "for_of")
    num_against_of = sum(labels == "against_of")
    num_for_l=sum(labels == "for_left" )
    num_for_r=sum(labels == "for_right")
    num_target_ob = num_for_l+num_for_r
    of_responses = num_for_of + num_against_of
    if of_responses == 0:
        oi = np.nan
    else:
        oi = (sum(labels == "for_of") - sum(labels == "against_of")) / of_responses
    if labels.shape[0] == 1:
        pi = np.nan
        pi_follow_of_only = np.nan
        p_left_right=np.nan
    else:
        pi = (of_responses - num_target_ob) / (of_responses + num_target_ob)
        pi_follow_of_only = (num_for_of - num_target_ob) / (num_for_of + num_target_ob)
        p_left_right=(num_for_l - num_for_r) / (num_for_l + num_for_r)
    

    return labels, oi, pi, pi_follow_of_only,p_left_right


def load_data(this_dir, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    locust_pattern = f"VR2*XY.h5"
    found_result = find_file(Path(this_dir), locust_pattern)
    dfxy = pd.read_hdf(found_result)
    dfxy["VR"] = np.tile(f"VR2", (len(dfxy), 1))
    dfxy["VR"] = dfxy["VR"] + "_" + dfxy["fname"]
    summary_pattern = f"VR2*score.h5"
    found_result = find_file(Path(this_dir), summary_pattern)
    df = pd.read_hdf(found_result)
    df["VR"] = np.tile(f"VR2", (len(df), 1))
    df["VR"] = df["VR"] + "_" + df["fname"]
    for key, grp in dfxy.groupby("fname"):
        # if key != "2024-10-14_144015":
        #     continue
        this_mu = grp["mu"].unique()
        if grp["heading"].shape[0] > 1:
            l, _, _, rotated_angles,_ = classify_heading_direction(
                grp["heading"].values, this_mu
            )
            fig2, (ax1, ax2) = plt.subplots(
                nrows=1, ncols=2, figsize=(18, 6), tight_layout=True
            )
            theta = np.radians(
                this_mu - 360
            )  # applying rotation matrix to rotate the coordinates
            # includes a minus because the radian circle is clockwise in Unity, so 45 degree should be used as -45 degree in the regular radian circle
            rot_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            rXY = np.column_stack((grp["X"].values, grp["Y"].values)) @ rot_matrix.T

            rXY_diff = np.diff(rXY, axis=1)
            test = np.arctan2(rXY_diff[0, 0, :][1], rXY_diff[0, 0, :][0])
            ax1.plot(rXY.T[0], rXY.T[1], color="k", linewidth=1)
            ax1.set(xlim=(-250, 250), ylim=(-250, 250), aspect=("equal"))
            xy = np.column_stack((grp["X"].values, grp["Y"].values))
            seg_no = 1
            for start, stop in zip(xy[:-1], xy[1:]):
                x, y = zip(start, stop)
                if l[seg_no] == "for_of":
                    this_color = "b"
                elif l[seg_no] == "target_ob":
                    this_color = "r"
                elif l[seg_no] == "against_of":
                    this_color = "c"
                else:
                    this_color = "k"
                ax2.plot(x, y, color=this_color, linewidth=1)
                ax2.set(xlim=(-250, 250), ylim=(-250, 250), aspect=("equal"))
                seg_no = seg_no + 1
            plt.show()


if __name__ == "__main__":
    thisDir = r"D:\MatrexVR_2024_Data\RunData\20250523_143428"
    json_file = "./analysis_methods_dictionary.json"
    tic = time.perf_counter()
    load_data(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
