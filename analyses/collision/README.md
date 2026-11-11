# Collision experiment

## Purpose

This workflow analyses the locomotor response to an approaching virtual stimulus. Its primary outcome is the latency until sustained slowing before the predicted collision. Trials without a detected stop remain in the analysis as right-censored observations.

The current design contains black and gregarious stimuli at constant speeds of 1 and 4.

## Files and execution order

1. `raw_data_preparation_collision.ipynb`

   Run this only when new recording folders need curated HDF5 files. It finds raw LocustVR `.dat` sessions and uses the shared `locustvr_extractor.py` code to create files such as `XY_full.h5` and `summary_full.h5`.

2. `data_preparation_collision.ipynb`

   Loads the curated HDF5 files, selects Collision animals from `GN25074` through `GN25118`, parses labels such as `Texture1_CS4`, aligns trajectories and writes the combined `locustvr_data.pkl` dataset.

3. `time_series_collision.ipynb`

   Creates frame-level speed data, detects sustained stops, exports the censored analysis table and runs the statistical models. The notebook also contains older optional deceleration, distance and time-course sections after the primary censored analysis.

4. `censored_statistics.py`

   Implements the repeated-measures statistical analysis. It is called by `time_series_collision.ipynb` and normally does not need to be run separately.

## Stop and censoring definition

- A stop is detected when speed falls below `0.2` and remains below that threshold for at least `0.5` seconds.
- Stop detection begins `0.5` seconds before stimulus onset.
- The predicted collision time is 20 seconds at speed 1 and 5 seconds at speed 4.
- If no qualifying stop occurs before collision, the trial is retained as right-censored.
- Incomplete frames, missing response windows, insufficient confirmation time, non-finite speed values and stops before stimulus onset are excluded with an explicit status reason.

## Statistical models

The hierarchical right-censored LogNormal analysis compares five hypotheses:

- null: no colour or speed effect
- speed: speed effect only
- colour: colour effect only
- additive: independent colour and speed effects
- interaction: colour-by-speed interaction

Repeated trials are grouped by animal, and trial order is included when available. Candidate Bayesian models are compared using PSIS-LOO. Effect tables report multiplicative latency ratios: values below 1 indicate earlier slowing for the numerator condition, and values above 1 indicate later slowing.

## Inputs and outputs

The mounted raw-data directory is:

```text
/Volumes/AG_Couzin-Fuchs/DATA/experiment_trackball_Optomotor/locustVR
```

The workflow writes generated analysis files below `results/`:

- `out_censored.csv`: valid observed and right-censored trials
- `out_censored_invalid_trials.csv`: excluded trials and their reasons
- `locust_censored_results/`: model summaries, comparisons, effect ratios and figures

The existing files in `results/locust_censored_results/` come from an earlier completed run. Their metadata deliberately retain the original input path as provenance.

## Important settings

Settings are stored in `analysis_methods_collision.json`.

The current setting `"overwrite_curated_dataset": true` allows raw-data preparation to rebuild existing curated HDF5 files. Change this to `false` when only newly added sessions should be processed.

Animal exclusion is configured with:

```json
"exclude_animals_marked_for_exclusion": true,
"exclude_molting_after_24h": false,
"exclude_molting_after_48h": false
```

Animals marked `T` in `Excluding this animal from analysis (Usually when animals die or molt, T/F)` are removed by default. The two molting options independently apply the 24- and cumulative 48-hour Google Sheet columns. Set an option to `false` to reverse that filter; empty cells and `F` remain included.
