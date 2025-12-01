import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import numpy as np

def projected_area(L, z):
    """
    Computes projected area of a looming flat object at distance z.

    Parameters:
    - L: diameter of the object (meters)
    - z: distance to the observer (meters)

    Returns:
    - A: projected area (arbitrary units, e.g., square meters)
    """
    A = (np.pi * L**2) / (4 * z**2)
    return A

def func(x, a, b,c):
    return a/np.tan(b*x)+c
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b
class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def calculate_expansion_rate(L, z, v):
    """
    Calculate angular expansion velocity of a looming stimulus.
    
    Parameters:
    - L: physical size of the object (meters)
    - z: distance to the object (meters)
    - v: approach velocity of the object (m/s), should be positive if the object is approaching
    
    Returns:
    - dtheta_dt: angular expansion velocity (radians per second)
    """
    numerator = L * v
    denominator = z**2 + (L / 2)**2
    dtheta_dt = numerator / denominator
    #v_linear = L * dtheta_dt /2
    v_linear = z * dtheta_dt
    return dtheta_dt,v_linear

# Example usage:
colormap_name = "Greys"
COL = MplColorHelper(colormap_name, 0, 5)
threshold_degree=40
L = 2     # Object diameter in centimeters
z = 10    # Distance to object in centimeters
v = 4     # Approach velocity in centimeters per second
fig, ax = plt.subplots(figsize=(5,5), dpi=250) 
use_angular_expansion_velocity=False
for this_speed in range(1,5):
    v_linear_list=[]
    v_angular_list=[]
    this_distance_list=[]
    area_list=[]
    for this_distance in range(1,42,3):
    #for this_distance in range(1,100):
        #print(this_distance)
        angular_velocity,v_linear = calculate_expansion_rate(L, this_distance, this_speed)
        p_area = projected_area(L, this_distance)
        angular_velocity_degree=np.degrees(angular_velocity)
        print(f"Angular expansion velocity: {angular_velocity:.6f} rad/s")
        print(f"Angular expansion velocity: {angular_velocity_degree:.6f} degree/s")
        print(f"linear expansion velocity: {v_linear:.6f} m/s")
        area_list.append(p_area)
        #ax.scatter(v_linear,this_distance)
        v_linear_list.append(v_linear)
        v_angular_list.append(angular_velocity_degree)
        this_distance_list.append(this_distance)

    if use_angular_expansion_velocity:
        these_v=np.vstack(v_angular_list)
    else:
        these_v=np.vstack(v_linear_list)

    these_distances=np.vstack(this_distance_list)
    #these_areas=np.vstack(area_list)
    ax.plot(these_v,these_distances,c=COL.get_rgb(this_speed),label=f'{this_speed} cm/s',linewidth=3)
    #ax.plot(these_areas,these_distances,c='r',linewidth=1)
     # start with values near those we expect
    x0=np.transpose(these_v)
    y0=np.transpose(these_distances)
    #s = np.tan(y0)
    #p0 = (1, 100,1)
    #params, cv = scipy.optimize.curve_fit(func, x0[0,:], y0[0,:], p0)
    p0 = (50, 0.5, 0.01)
    params, cv = scipy.optimize.curve_fit(monoExp, x0[0,:], y0[0,:], p0)
    m, t, b = params
    #a, b,c = params
    #ax.plot(x0[0,:], func(x0[0,:], m, t,b),c='b',linewidth=1)
    #ax.plot(s, y0,c='r',linewidth=1)
    print(params)
if use_angular_expansion_velocity:
    xlabel_text = "expansion/contraction speed (degree/s)"
    f_name=f"angular_expansion_speed_object_{L}cm_speed"
else:
    xlabel_text = "expansion/contraction speed(cm/s)"
    f_name=f"linear_expansion_speed_object_{L}cm_speed"
threshold_degree_line=L/2/np.tan(np.radians(threshold_degree)/2)
ax.hlines(threshold_degree_line, 0, 4, colors='r', linestyles='--')
ax.set(
    xlabel=xlabel_text,
    ylabel="distance from the test animal (cm)",
    yticks=([0,2,20,40]),
    #xticks=([0,2,4]),
)
ax.legend(title="agent speed",loc='upper right')
fig_name=f"{f_name}.svg"
fig.savefig(fig_name)
fig_name=f"{f_name}.png"
fig.savefig(fig_name)
# linear expansion velocity: 0.216058 m/s
# Angular expansion velocity: 0.004997 rad/s
# Angular expansion velocity: 0.286300 degree/s
# linear expansion velocity: 0.199875 m/s
# [70.04534966  3.58192544  3.94627183]