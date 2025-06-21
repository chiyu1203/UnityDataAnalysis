import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experiment_name": "choice",
    "overwrite_curated_dataset": True,
    "export_fictrac_data_only":False,
    "save_output": False,
    "time_series_analysis": True,
    "filtering_method": "sg_filter",
    "plotting_trajectory": False,
    "plotting_event_distribution": True,
    "distribution_with_entire_body": True,
    "load_individual_data": True,
    "select_animals_by_condition": True,
    "active_trials_only": True,
    "split_stationary_moving_ISI":True,
    "align_with_isi_onset": False,
    "extract_follow_epoches": True,
    "follow_locustVR_criteria": True,
    "calculate_follow_chance_level": True,
    "frequency_based_preference_index":True,
    "analyse_first_half_only":True,
    "analyse_second_half_only":False,
    "exclude_extreme_index":False,
    "graph_colour_code": ["r", "b", "g", "k", "c", "y", "m", "r"],
    "follow_within_distance": 50,
    "camera_fps": 100,
    "trackball_radius_cm": 5,
    "monitor_fps": 60,
    "body_length": 4,
    "growth_condition": "G",
    "analysis_window": [-10, 10],
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
