import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "swarm",
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "debug_mode": False,
    "fictrac_posthoc_analysis": True,
    "use_led_to_align_stimulus_timing": True,
    "align_with_isi_onset": False,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": True,
    "plotting_event_related_trajectory": False,
    "plotting_deceleration_accerleration": False,
    "plotting_optomotor_response": True,
    "generate_locust_vr_matrices": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": False,
    "camera_fps": 120,
    "trackball_radius": 50,
    "monitor_fps": 60,
    "body_length": 0.12,
    "growth_condition": "G",
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
