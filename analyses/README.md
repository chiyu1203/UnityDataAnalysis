# Experiment-specific analyses

The repository contains two independent analysis workflows that share the extraction and utility code in the repository root.

## Collision experiment

Run the notebooks in this order:

1. `collision/raw_data_preparation_collision.ipynb` when new raw sessions need HDF5 files.
2. `collision/data_preparation_collision.ipynb` to build the combined collision dataset.
3. `collision/time_series_collision.ipynb` to detect sustained stopping and run the censored statistical analysis.

The statistical model is implemented in `collision/censored_statistics.py`. Generated tables and plots are written below `collision/results/`.

## Colour-preference experiment

Run the notebooks in this order:

1. `color_preference/raw_data_preparation_color_preference.ipynb` when new raw sessions need HDF5 files.
2. `color_preference/data_preparation_color_preference.ipynb` to apply spreadsheet exclusions and build the combined dataset.
3. `color_preference/visualising_color_choice.ipynb` to produce the colour-choice figures and summary tables.

Each workflow has its own `analysis_methods_*.json` file. This prevents settings for one experiment from changing the other experiment.

The older branches remain available as backups, but day-to-day work can now be performed from this single branch.
