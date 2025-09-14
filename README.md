# Calisma

Welcome to the Calisma Project repository.  This repository contains code and resources for training and evaluating machine learning models for predicting climate variables, specifically Cloud Fraction (CLF) and Liquid Water Path (LWP).

## Environment Setup

First, create the conda environment and install dependencies by running the following script:</br>
(you should have conda and python installed to run this) </br>
`conda env create -f environment.yaml` 

Then, activate the newly created environment using:</br>
`conda activate calisma`</br>
The environment should be activated in your terminal. To run the code using your application UI select this [conda environment as your python interpreter.](https://www.google.com/search?q=select+existing+conda+environment+as+python+interpreter&client=ubuntu-sn&hs=rkJ&sca_esv=63c9e37f8da915f5&channel=fs&sxsrf=AHTn8zqxaLTBEhH0yx9c5RTSDwiCU5Jqzw%3A1742553270476&ei=tkDdZ87cHIWui-gP_4WP6QE&ved=0ahUKEwjOj8Hl_JqMAxUF1wIHHf_CIx0Q4dUDCBA&uact=5&oq=select+existing+conda+environment+as+python+interpreter&gs_lp=Egxnd3Mtd2l6LXNlcnAiN3NlbGVjdCBleGlzdGluZyBjb25kYSBlbnZpcm9ubWVudCBhcyBweXRob24gaW50ZXJwcmV0ZXIyBRAAGO8FMgUQABjvBTIFEAAY7wUyBRAAGO8FMggQABiiBBiJBUiHGVCzBVipEnABeAGQAQCYAYYBoAGiBqoBAzYuM7gBA8gBAPgBAZgCCaACtgbCAgoQABiwAxjWBBhHwgIHECMYsAIYJ8ICCBAAGIAEGKIEmAMAiAYBkAYIkgcDNS40oAe4ObIHAzQuNLgHrAY&sclient=gws-wiz-serp)

## Contributing to the repository

To contribute to this project, create a new branch using: </br>
`git checkout -b <branch_name>` </br>
Commit and push your edits to this new branch. And then create a pull request on Github.

## Code Walkthrough:

### Model Training and Analyses
[multimodal_train_and_evaluate.ipynb](multimodal_train_and_evaluate.ipynb) is the most important notebook to run the training, evaluation and analyses.
The `predictands` variable name should be set to '_both_' for multi-target prediction and to _'clf'_ or _'lwp'_ for respective single-target predictions (more details in the code comments).

### Below are Optional Files if you want to do more analyses

###  Visual Data Analyses (Pre-training)
The Jupyter notebook [pre_plot_analyses.ipynb](pre_plot_analyses.ipynb) performs following tasks:
- Data conversion (to Xarrays for better visualisation)
- Data statistics (mean, std, etc.)
- Map temporal and spatial distribution of variables

###  Data Preprocessing
The script [preprocessing.py](preprocessing.py) contains functions to
- Remove Outliers
- Scale the dataset 
- Compute correlation
- Split using a fixed time value </br>

###  Single-target Training and Evaluation 
- The script [train_and_evaluate.py](train_and_evaluate.py) handles single-target training, evaluation, and related model functions.
- The script [train_evaluate_analyses.ipynb](train_evaluate_analyses.ipynb) - Evaluate the results of training models and some visual plots




