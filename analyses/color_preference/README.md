# Colour-preference experiment

## Purpose

This workflow analyses choices between black, yellow and gregarious visual textures. It supports the three mixed texture pairs and the three identical-texture controls. Presentation side and bait texture remain available as separate columns, but trials are not grouped or sorted by bait texture.

## Files and execution order

1. `raw_data_preparation_color_preference.ipynb`

   Run this only when new recording folders need curated HDF5 files. It finds raw LocustVR `.dat` sessions and uses the shared `locustvr_extractor.py` code to create files such as `XY_full.h5` and `summary_full.h5`.

2. `data_preparation_color_preference.ipynb`

   Loads curated `XY_full.h5` files, reads animal metadata from the LocustVR Google Sheet, applies per-animal trial cutoffs, parses the extended trial labels and writes `locustvr_data.pkl`.

3. `visualising_color_choice.ipynb`

   Classifies choices and creates the trial overview, pairwise preference plot, choice composition, texture-aligned trajectories and optional side-bias control.

## Trial labels

The expected format is:

```text
T0_CD6_CS1_S1_Lyellow_Rgregarious_BTblack
```

The parser extracts:

- `distance_cm`
- `condition_code`
- `label_state`
- `left_texture`
- `right_texture`
- `bait_texture`

The original `trial_label` is retained unchanged.

## Animal and trial selection

- The spreadsheet column `Exclude trials after trial X:` defines an inclusive final trial for each animal. Later trials are excluded.
- An empty cutoff cell keeps all available trials for that animal.
- Invalid, negative or non-integer cutoff values stop the notebook with an error.
- Animals marked `T` in `Excluding this animal from analysis (Usually when animals die or molt, T/F)` are excluded by default. Set `exclude_animals_marked_for_exclusion` to `false` only when those animals should deliberately be restored.
- Molting exclusion is controlled independently by `exclude_molting_after_24h` and `exclude_molting_after_48h`. The corresponding Google Sheet columns are applied even if the other individual-selection options are disabled.
- The current notebook line `dir_lis_2026 = dir_list[-19:]` restricts loading to the final 19 discovered session folders. Review this line whenever animals are added or the folder order changes.

## Choice definition

- The analysis window is 0–20 seconds of the test phase.
- A trial is classified from its mean heading during this window.
- Mean headings between −75° and −45° are classified as left choices.
- Mean headings between +45° and +75° are classified as right choices.
- All other headings are classified as neutral.
- For mixed pairs, the selected texture is obtained from the original left/right presentation metadata.
- Identical-texture trials are shown in control summaries but are not used for pairwise preference indices.

## Side handling

`FLIP_SIDES = True` aligns mixed trials by texture, so the result describes texture preference independent of presentation side.

Set `FLIP_SIDES = False` to retain the original left/right orientation and inspect possible side bias.

## Inputs and outputs

The mounted raw-data directory is:

```text
/Volumes/AG_Couzin-Fuchs/DATA/experiment_trackball_Optomotor/locustVR
```

The combined pickle and generated figures are currently written to the `save_path` configured in the notebooks:

```text
/Users/jonny/Desktop/Uni/Master/collevtive behavior/VR_analysis/collor_choice
```

Main exported tables:

- `trial_texture_choices.csv`
- `pairwise_preference_indices.csv`

Main figures include the trial-choice matrix, pairwise preference overview, choice composition, texture-aligned trajectories and side-bias control.

## Important settings

Settings are stored in `analysis_methods_color_preference.json`.

The current setting `"overwrite_curated_dataset": false` preserves existing HDF5 files during raw-data preparation and processes only sessions without curated output.

Animal exclusion is configured with:

```json
"exclude_animals_marked_for_exclusion": true,
"exclude_molting_after_24h": false,
"exclude_molting_after_48h": false
```

The general exclusion flag is active by default. Enable either molting option when that time window should also be excluded. Because the 48-hour spreadsheet column is cumulative, its `T` values include animals already marked at 24 hours. Empty cells and `F` remain included.
