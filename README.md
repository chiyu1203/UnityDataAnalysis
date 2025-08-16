# Unity VR data analysis

This repo includes codes to analyse data collected matrex VR, a unity-based VR system for insects (https://github.com/pvnkmrksk/matrexVR)

## Data base structure
The default structure comes the Unity program is good. However, there were slight differences between Scenes and how agent's location is logged.

## Install Hdf5view

Since Sercan manages his dataset in hdf file. Most of the curated dataset created by this repo is stored in that format too.

The easiest way to view hdf file without coding is to install Hdf5 viwer, which can be found here <https://www.hdfgroup.org/downloads/hdfview/>

In Windows PC, the file to download is something like HDFView-3.1.3-win10_64-vs16.zip

### Install Visual Studio Code

This is optional but in case some functions interact with the host pc is editor-dependant. I would suggest to use Visual Studio Code.

After installing the VS code, remember to install extension on Visual Studio Code: 

press *ctrl + shift + x* or click on Extensions when you are in the VS code window, install _Python_, _Code Runner_, *Code Spell Checker*.

The following extension are optional depends on your need

C#: useful to create customise bonsai node or workflow

Github copilot: a good helper for coding

Remote SSH: if you want to remote in the PC in the lab from your own PC somewhere else (doable, but it needs MPIAB VPN and some additional SSH configuration)

If you want to use VCC's GPU cluster, you need to install _WSL_.

Note that WSL does not share all of the extensions from Windows so you need to check if abovementioned extensions are installed

Then inside the WSL environment, you should install _Kubernetes_

Lastly some formatter would be helpful but this is optional. I use _Black Formatter_ , _Rainbow CSV_


### Install Git

Use the default setting to install git and create a folder called *GitHub* (at Documents, this will be handy later)


### Clone this repository on Visual Studio Code

On this page, click *Code* above the repository and then At the *Clone* section, copy the https path.

On the VS code window, go to *source control* and then click *clone repository* and paste the https path there and press enter

Cloning the main branch and create Pull Request if needed.

## Set up python environment

General anaconda environment for all setups and data analyses

Install the latest anaconda version for your operating system (https://www.anaconda.com/products/individual).

Open the anaconda prompt and create a virtual environment via conda

## creating virtual environment and installing packages via commands

Below is the command to install packages via conda environment 

```
conda create --name unity_analysis --channel conda-forge python=3.11
conda activate unity_analysis
conda update -n base -c defaults conda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install --yes -c conda-forge -v h5py opencv ipython jupyter matplotlib pandas matplotlib scipy jupyterlab seaborn ipyparallel pytables deepdiff pyarrow fastparquet
```
[Optional] if you ever want to load database from an excel sheet
```
conda install openpyxl
```

If environmental.yaml or requirements.txt works, try creating the virtual environment with the file.
```
conda env create -f environment.yml
conda create --name unity_analysis --file requirements.txt
```
It seems that environment.yml is better for environment with packages installed via conda....

## Other modules required to run this analysis

There are useful tools stored in another repository so after you cloned UnityDataAnalaysis, remember to clone utilities as well and place them all under the GitHub folder

```
git clone https://github.com/chiyu1203/utilities.git
```
[Optional] to export the environment for other PCs, try the following commands
```
conda env export > environment.yml --no-builds
conda list -e > requirements.txt
```
# Start coding

## How to use the analysis pipeline and jupyter notebook

Use **data_exploration_tutorial.ipynb**, which includes the essence of **locustvr_converter.py** in this repo and some pilot plotting functions.

The analysis methods are organised into a json file, which is created from a python file. This json file tells the notebook and python script how to analysis the data (with some boolean options) and add meta information of the experiment that is not included during data collection.

Therefore, each project has its own json file. Below explains what those analysis methods are about.

    "experiment_name": "choice", this means Scene name in the unity programme. The main difference between scene is nevertheless whether how agent's position is logged. Any open-loop (agent's movement is not locked to focal animal's movement) experiment uses "choice". Any closed-loop experiments should use "band". For the data coming from locustvr, fill in "locustvr"

    "overwrite_curated_dataset": boolean, whether to delete the existing HDF file or not. If True, **locustvr_converter.py** or **locustvr_extractor.py** will delete the old curated dataset before processing raw data.

    "export_fictrac_data_only": this is used in locustVR_converter and data_exploration notebook. This option will skip the rest of processing in locustVR_converter after exporting the fictrac raw data into a parquet file format. 

    "save_output": boolean, whether to save any output (including dataset and figures) during data analysis. If True, then save any output

    "time_series_analysis": boolean, analyse where animals move at every time points in the experiment. If True, select a filter method to remove tracking noise, If false, spatial discretisation based on nymph's body size. will be applied to quantify animal's trajectory.
    
    "filtering_method": 'sg_filter', default is to use savgol_filter. Feel free to add your own filter of interest. At the moment, any string other than 'sg_filter' will lead to no filter. 
    
    "plotting_trajectory": boolean, whether to plot the trajectory of the animals in the experiment.

    "plotting_event_distribution": boolean, whether to plot the distribution of follow epochs across trials in an experiment

    "distribution_with_entire_body": boolean, whether to consider the entire body of agents when plotting the 2D heatmap. If true, a 2x6 cm rectangle will be created around the centriod for the heatmap. If false, use centroid (the 0x0x0 point in the Blender model, which is 4 cm away from the most posterior and 2cm away from the most anterior tip)

    "load_individual_data" and "select_animals_by_condition": boolean, these are used in the jupyter notebook to load specific animals for analysis. If both are True, you need to specify what condition in a dictionary. If either of them is False, all animals in the database will be included.

    "active_trials_only": boolean, a helper argument to extract animals whose walking distance pass certain threshold. So far, this is only used in **data_exploration_tutorial.ipynb**

    "align_with_isi_onset": boolean, use together with "analysis_window" in **time_series_analysis_tutorial.ipynb**, if True, then 0 in "analysis_window" means the onset of inter-stimulus interval; if False, then 0 means the onset of stimuli.

    "extract_follow_epoches": boolean,this is used in "sorting_time_series_analysis.py" to extract trajectory during follow_epoches for further analysis. If false, the the entire trajectory durning the experiment will be extracted
    
    "follow_locustVR_criteria": boolean,this is used in "sorting_time_series_analysis.py" to classify follow behaviour. If true, the programme will use distance, velocity and degree deviation to define follow behaviour. If false, the programme will use only distance and velocity.

    "calculate_follow_chance_level": boolean, work in progress, this is used to calculate the frequency of follow behaviour happens at the chance level. Current plan is to shuffle the position of agents across trials. This, however, may violate the trial exchangeability assumption depending on experiment design.

    "frequency_based_preference_index": boolean, this is used in the preference assay. Instead of calculating the time of follow epochs, use the frequency of entering a follow epoch.

    "analyse_first_half_only" and "analyse_second_half_only": boolean. This is used in the sorting_time_series_analysis.py. Default settings: both of them is false to analyse the entire experiment. There is no definition for both of them is true.

    "exclude_extreme_index": boolean. This is used when calculating preference index. This will include animals that only choose one type of option in the experiment, which results in 1 or -1, a rather extreme preference.
    
    "graph_colour_code": array of string, just a helper array to know which colour is used when plotting the data.

    "follow_within_distance": int, one of the criteria to define follow behaviour in "sorting_time_series_analysis.py"

    "camera_fps": 100, #default video acqusition rate in matrexVR
    
    "trackball_radius_cm": 0.5,#default size of air-ball in matrexVR
    
    "monitor_fps": 60,#default monitor target render frequency in matrexVR
    
    "body_length": 4, #Unit: cm default body length use in spatial discretisation.
    
    "growth_condition": "G", Note: "G" for gregarious "S" for solitarious animals

    "analysis_window": [-10,10], a two-element array to define where to slice trials (0 means the onset, unit: sec) in **time_series_analysis_tutorial.ipynb**

Use **time_series_analysis.ipynb**, if you want to analyse stimulus-evoked responses in details.

Use **preference_analysis.ipynb**, if you want to calculate animal's preference index based on follow behaviour.




