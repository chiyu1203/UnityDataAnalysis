# Unity VR data analysis

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

Github copilot: sounds useful but never used it

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
conda install --yes -c conda-forge -v h5py opencv ipython jupyter matplotlib pandas matplotlib scipy jupyterlab seaborn ipyparallel pytables
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

    "experiment_name": "choice", this means Scene name in the unity programme, which is used to check datasets from different logging system.

    "overwrite_curated_dataset": boolean, whether to overwrite the existing HDF file or not. If True, delete the old curated dataset.

    "dont_save_output": boolean, whether to save any output (including dataset and figures) during data analysis. If True, then do not save any output

    "time_series_analysis": boolean, analyse where animals move across trials. If True, select a fileter method to remove tracking noise, If false, spatial discretisation will be applied to analyse animal's trajectory.
    
    "filtering_method": 'sg_filter', default is to use savgol_filter. Any string other than that will lead to no filter. 
    
    "plotting_trajectory": boolean, whether to plot the trajectory of the animals in the experiment.
    
    "plotting_event_related_trajectory": boolean, whether to plot a heat map of animal's trajectory after the stimulus onset
    
    "plotting_deceleration_accerleration": boolean, whether to plot average deceleration and accerleration onset of the animals (this is still under construction)

    "load_individual_data" and "select_animals_by_condition": boolean, these are used in the jypter notebook to extract specific animals for analysis. If both are True, you need to specify what condition in a dictionary. If either of them is False, all animals in the database will be included.
    
    "graph_colour_code": array of string, just a helper array to know which colour is used.

    "camera_fps": 100, #default video acqusition rate in matrexVR
    
    "trackball_radius_cm": 0.5,#default video acqusition rate in matrexVR
    
    "monitor_fps": 60,#default monitor target render frequency in matrexVR
    
    "body_length": 4, #Unit: cm default body length use in spatial discretisation.
    
    "growth_condition": "G", Note: "G" for gregarious "S" for solitarious animals

Use **time_series_analysis.ipynb**, if you want do analyse stimulus-evoked responses in details.




