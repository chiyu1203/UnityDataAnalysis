# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:22:37 2023

@author: Sercan
"""

import os
import time
import math
from math import atan2
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from threading import Lock

from funcs import *

# Initialize lock
lock = Lock()

# Constants
BODY_LENGTH = 0.12

FOLDER = "Z:/Users/chiyu/Data"
SAVE_FOLDER = "Z:/Users/chiyu/Output"

# Get subfolders
subfolders = [f.name for f in os.scandir(FOLDER) if f.is_dir()]
print(subfolders)
# Initialize lists
density = []
order = []
angle = []
files = []
# Process subfolders
for i in subfolders:
    n = i.split("_")
    if len(n) == 7:
        density.append(int(float(n[3])))
        order.append(float(n[4]))
        angle.append(float(n[5]))
        files.append(i)

# Map densities to strings
re_density = [str(i).zfill(2) for i in density]

# Create groups
groups = [f"{y}_{x}" for x, y in zip(re_density, order)]

print(len(files))


# def process_file(file, i):
def process_file(file, i):
    print(i)
    print(file)
    b1 = os.path.join(FOLDER, file, "velocities.dat")
    b2 = os.path.join(FOLDER, file, "velocities_all.dat")

    for basepath in [b1, b2]:
        if os.path.isfile(basepath):
            # print(basepath)
            df = pd.read_csv(basepath, sep=" ", header=0)
            if "velocities_all" in basepath:
                df = df.iloc[:, -4:-2]
                df = df.cumsum() * -1
            X = df.iloc[:, 0].to_numpy()
            Y = df.iloc[:, 1].to_numpy()
            del df

            loss, X, Y = removeNoiseVR(X, Y)
            loss = 1 - loss

            if len(X) == 0:
                print("all is noise")
                continue

            rX, rY = rotate_vector(X, Y, -angle[i])

            # fig = plt.figure()
            # plt.plot(rX, rY)
            # plt.ylim(-30, 30)
            # plt.xlim(-30, 30)
            # plt.title(file)
            # plt.savefig(os.path.join(SAVE_FOLDER, f"{file}.png"))
            # plt.close(fig)

            newindex = diskretize(rX, rY, BODY_LENGTH)
            dX = np.array([rX[i] for i in newindex]).T
            dY = np.array([rY[i] for i in newindex]).T

            chop = file.split("_")[:2]
            fchop = "_".join(chop)

            angles = np.array(ListAngles(dX, dY))

            c = np.cos(angles)
            s = np.sin(angles)

            xm = np.sum(c) / len(angles)
            ym = np.sum(s) / len(angles)

            meanAngle = atan2(ym, xm)
            meanVector = np.sqrt(np.square(np.sum(c)) + np.square(np.sum(s))) / len(
                angles
            )

            std = np.sqrt(2 * (1 - meanVector))

            tdist = len(dX) * BODY_LENGTH

            sin = meanVector * np.sin(meanAngle)
            cos = meanVector * np.cos(meanAngle)

            f = [fchop] * len(dX)
            loss = [loss] * len(dX)
            o = [order[i]] * len(dX)
            d = [density[i]] * len(dX)
            G = [groups[i]] * len(dX)

            dfXY = pd.DataFrame(
                {
                    "X": dX,
                    "Y": dY,
                    "fname": f,
                    "loss": loss,
                    "order": o,
                    "density": d,
                    "groups": G,
                }
            )

            f = [f[0]]
            loss = [loss[0]]
            o = [o[0]]
            d = [d[0]]
            G = [G[0]]
            V = [meanVector]
            S = [meanAngle]
            ST = [std]
            lX = [dX[-1]]
            tD = [tdist]
            sins = [sin]
            coss = [cos]

            df = pd.DataFrame(
                {
                    "fname": f,
                    "loss": loss,
                    "order": o,
                    "density": d,
                    "groups": G,
                    "score": S,
                    "vector": V,
                    "variance": ST,
                    "distX": lX,
                    "distTotal": tD,
                    "sin": sins,
                    "cos": coss,
                }
            )
            print(df)

            # Use lock to prevent concurrent writes
            with lock:
                store = pd.HDFStore(os.path.join(SAVE_FOLDER, "XY.h5"))
                store.append(
                    "name_of_frame", dfXY, format="t", data_columns=dfXY.columns
                )
                store.close()

                store = pd.HDFStore(os.path.join(SAVE_FOLDER, "score.h5"))
                store.append("name_of_frame", df, format="t", data_columns=df.columns)
                store.close()


# Process files in parallel
# with ThreadPoolExecutor(max_workers=6) as executor:
# executor.map(process_file, files, range(len(files)))


# for count, f in enumerate(files):
# process_file(f, count)
