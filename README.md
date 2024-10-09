# DataLiteracyProject

## Project Description
Breast cancer represents the most prevalent manifestation of cancer in women.
This project investigates age-related inequalities in treatment selection and efficacy of chemotherapy in 1304 patients from the Molecular Taxonomy of Breast Cancer International Consortium database. 

This Project is part of the WS23/24 Data Literacy course @Eberhard Karls University Tuebingen by Prof. Dr. Hennning (ML4201).

---

## Results
We use multiple logistic regression models for predicting the type of surgery, chemotherapy, radio therapy, and hormone therapy. Kaplan-Meier analysis compared overall and relapse survival of breast cancer patients within the same tumor grade, distinguishing those aged 50 and older from those younger than 50. A log-rank test assessed differences in relapse-free and overall survival between these age groups. Although older patients undergo chemotherapy significantly less frequently, the recurrence rate and overall survival status remain comparable to those of younger patients within the same tumor stage.

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
conda create -n brca python=3.9 ipykernel ipywidgets=7
conda activate brca

# install ipython kernel
ipython kernel install --user --name=brca

# install requirements
pip install -r requirements.txt
````

--- 

## How to reproduce our results
To reproduce the results of our analysis go to the scr folder and run the corresponding notebooks.

To reproduce our plots go to doc/fig and run the corresponding notebooks.
