# Austin Coursey and Abel Diaz-Gonzalez AML Project

## Setup

Create and activate the conda environment using:

`conda env create -f environment.yml`

`conda activate aml`

Attach the environment to Jupyter using:

`python -m ipykernel install --user --name=aml`

Open this notebook using the command

`jupyter notebook`

inside your activated aml environment (if this command does not work, use pip install notebook.)

Go to Kernel -> change kernel -> aml at the top.

Download the data from https://vanderbilt.box.com/s/ljzx9zjcrlx1re9bfh331nl743k1shxz and put it in this directory.

## Relevant Code

Code showing how we train normalizing flow models and perform empirical Bayes is shown in `showcase.ipynb`. Code used to train models on the Tennessee Eastman Process dataset is shown in `TEP.ipynb`. Code used to run the hyperparameter grid search is shown in `hpo.py`.
