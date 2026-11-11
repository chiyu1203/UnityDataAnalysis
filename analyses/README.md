# Experiment analyses

This directory contains the two maintained LocustVR analysis workflows. Both workflows use the shared extraction and utility code in the repository root, but keep their notebooks, settings and outputs separate.

## Collision experiment

The Collision workflow measures when a locust slows or stops while a virtual stimulus approaches at different speeds and with different textures.

See [`collision/README.md`](collision/README.md) for the execution order, stop definition, censoring rules and statistical outputs.

## Colour-preference experiment

The Colour Preference workflow compares choices between black, yellow and gregarious textures, including identical-texture control trials and an optional left/right side-bias view.

See [`color_preference/README.md`](color_preference/README.md) for the execution order, trial-label format, exclusion rules and figure definitions.

## Shared code

The experiment folders import the following shared components from the repository root:

- `locustvr_extractor.py`
- `LocustVR_data_analysis/utils_local.py`
- the sibling `utilities` directory

Each workflow has its own `analysis_methods_*.json` file. Changing one experiment's settings therefore does not change the other experiment.
