# DataLiteracyProject

Project for the “Data Literacy” course at the University of Tübingen, Winter 2023/24 (Module ML4201).

---

## Setup
Make sure you have Python and Git installed on your system.
### Clone the repository
````shell
git clone https://github.com/mawa00006/DataLiteracyProject.git
cd DataLiteracyProject
````
### Create a conda environment and install all necessary dependencies
````shell
# create a conda environment with python 3.9 and ipykernel
conda create -n ProjectName python=3.9 ipykernel ipywidgets=7
conda activate ProjectName

# install ipython kernel
ipython kernel install --user --name=ProjectName

# install requirements
pip install -r requirements.txt
````

--- 

## How to reproduce our results
To reproduce the results of our analysis go to the scr folder and run the corresponding notebooks.

To reproduce our plots go to doc/fig and run the corresponding notebooks.
