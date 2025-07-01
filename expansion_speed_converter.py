import numpy as np
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
L = 3     # Object diameter in centimeters
z = 10    # Distance to object in centimeters
v = 4     # Approach velocity in centimeters per second
fig, ax = plt.subplots(figsize=(5,5), dpi=250) 
use_angular_expansion_velocity=True
for this_speed in range(1,5):
    v_linear_list=[]
    v_angular_list=[]
    this_distance_list=[]
    for this_distance in range(1,42,3):
#    for this_distance in range(1,40):
        #print(this_distance)
        angular_velocity,v_linear = calculate_expansion_rate(L, this_distance, this_speed)
        angular_velocity_degree=np.degrees(angular_velocity)
        print(f"Angular expansion velocity: {angular_velocity:.6f} rad/s")
        print(f"Angular expansion velocity: {angular_velocity_degree:.6f} degree/s")
        print(f"linear expansion velocity: {v_linear:.6f} m/s")
        #ax.scatter(v_linear,this_distance)
        v_linear_list.append(v_linear)
        v_angular_list.append(angular_velocity_degree)
        this_distance_list.append(this_distance)
    if use_angular_expansion_velocity:
        these_v=np.vstack(v_angular_list)
    else:
        these_v=np.vstack(v_linear_list)
    these_distances=np.vstack(this_distance_list)
    ax.plot(these_v,these_distances,c=COL.get_rgb(this_speed),label=f'{this_speed} cm/s',linewidth=3)
if use_angular_expansion_velocity:
    xlabel_text = "expansion/contraction speed (degree/s)"
    f_name="angular_expansion_speed_object_speed"
else:
    xlabel_text = "expansion/contraction speed(cm/s)"
    f_name="linear_expansion_speed_object_speed"
ax.set(
    xlabel=xlabel_text,
    ylabel="distance from the test animal (cm)",
    yticks=([1,20,40]),
    #xticks=([0,2,4]),
)
ax.legend(title="agent speed",loc='upper right')
fig_name=f"{f_name}.svg"
fig.savefig(fig_name)
fig_name=f"{f_name}.png"
fig.savefig(fig_name)
