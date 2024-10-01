import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "choice",
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "b", "g", "k", "c", "y", "m", "r"],
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
    "load_individual_data": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": False,
    "time_series_analysis": True,
    "camera_fps": 100,
    "trackball_radius_cm": 0.5,
    "monitor_fps": 60,
    "body_length": 12,
    "growth_condition": "G",
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
