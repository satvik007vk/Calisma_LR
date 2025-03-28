# Calisma

Welcome to the Calisma Project repository. 

## Environment Setup

First, create the conda environment and install dependencies by running the following script:</br>
(you should have conda and python installed to run this) </br>
`conda env create -f environment.yml` 

Then, activate your environment using:</br>
`conda activate calisma`
Select this [conda environment as your python interpreter.](https://www.google.com/search?q=select+existing+conda+environment+as+python+interpreter&client=ubuntu-sn&hs=rkJ&sca_esv=63c9e37f8da915f5&channel=fs&sxsrf=AHTn8zqxaLTBEhH0yx9c5RTSDwiCU5Jqzw%3A1742553270476&ei=tkDdZ87cHIWui-gP_4WP6QE&ved=0ahUKEwjOj8Hl_JqMAxUF1wIHHf_CIx0Q4dUDCBA&uact=5&oq=select+existing+conda+environment+as+python+interpreter&gs_lp=Egxnd3Mtd2l6LXNlcnAiN3NlbGVjdCBleGlzdGluZyBjb25kYSBlbnZpcm9ubWVudCBhcyBweXRob24gaW50ZXJwcmV0ZXIyBRAAGO8FMgUQABjvBTIFEAAY7wUyBRAAGO8FMggQABiiBBiJBUiHGVCzBVipEnABeAGQAQCYAYYBoAGiBqoBAzYuM7gBA8gBAPgBAZgCCaACtgbCAgoQABiwAxjWBBhHwgIHECMYsAIYJ8ICCBAAGIAEGKIEmAMAiAYBkAYIkgcDNS40oAe4ObIHAzQuNLgHrAY&sclient=gws-wiz-serp)

## Code Walkthrough:

### 1) Visual Data Analyses (Pre-training)
The Jupyter notebook [pre_plot_analyses.ipynb](pre_plot_analyses.ipynb) performs following tasks:
- Data conversion (to Xarrays for better visualisation)
- Data statistics (mean, std, etc.)
- Map temporal and spatial distribution of variables

### 2) Data Preprocessing
The script [preprocessing.py](preprocessing.py) contains functions to
- Remove Outliers
- Scale the dataset 
- Compute correlation
- Split using a fixed time value </br>
To save the training and test datasets in your locally, remove comments from the last part of code.

### 3) Training and Evaluation
The script [train_and_evaluate.py](train_and_evaluate.py) contain the following parts: </br>
(Paths to the newly saved train and test datasets from step 2 needs to be updated for running further code) 
- Function to load train-test data and determine the predictors(X) and predictands(y). 
The predictand/s could be chosen as clf, lwp or both. The predictors remain fixed as all other variables.
- Functions to train different models.
- Function to evaluate the chosen model performance. 
- Some other functions specific to models (eg, get_mlr_coefficients)
- shap_explainability model (needs troubleshooting)

### 4) Visual Data Analyses (Post-training)
The script [train_evaluate_analyses.ipynb](train_evaluate_analyses.ipynb) calls the functions from other python scripts to </br>
- Evaluate the results of training models and some visual plots
- Some significant results are documented in [results.md](results.md) file.

## Contributing to the repository

To contribute to this project, create a new branch using: </br>
`git checkout -b <branch_name>` </br>
Commit and push your edits to this new branch. And then create a pull request on Github.
