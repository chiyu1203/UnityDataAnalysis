import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "choice",
    "overwrite_curated_dataset": True,
    "dont_save_output": False,
    "time_series_analysis": False,
    "filtering_method": "sg_filter",
    "plotting_trajectory": False,
    "plotting_deceleration_accerleration": False,
    "load_individual_data": True,
    "select_animals_by_condition": True,
    "graph_colour_code": ["r", "b", "g", "k", "c", "y", "m", "r"],
    "camera_fps": 100,
    "trackball_radius_cm": 0.5,
    "monitor_fps": 60,
    "body_length": 4,
    "growth_condition": "G",
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
