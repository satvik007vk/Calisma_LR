# Calisma

Welcome to the Calisma Project repository. 
Below are the instructions to run the analyses


First, create the conda environment and install dependencies by running the following script:</br>
`conda env create -f environment.yml` 

Then, activate your environment using:</br>
`conda activate calisma`
Select this [conda environment as your python interpreter.](https://www.google.com/search?q=select+existing+conda+environment+as+python+interpreter&client=ubuntu-sn&hs=rkJ&sca_esv=63c9e37f8da915f5&channel=fs&sxsrf=AHTn8zqxaLTBEhH0yx9c5RTSDwiCU5Jqzw%3A1742553270476&ei=tkDdZ87cHIWui-gP_4WP6QE&ved=0ahUKEwjOj8Hl_JqMAxUF1wIHHf_CIx0Q4dUDCBA&uact=5&oq=select+existing+conda+environment+as+python+interpreter&gs_lp=Egxnd3Mtd2l6LXNlcnAiN3NlbGVjdCBleGlzdGluZyBjb25kYSBlbnZpcm9ubWVudCBhcyBweXRob24gaW50ZXJwcmV0ZXIyBRAAGO8FMgUQABjvBTIFEAAY7wUyBRAAGO8FMggQABiiBBiJBUiHGVCzBVipEnABeAGQAQCYAYYBoAGiBqoBAzYuM7gBA8gBAPgBAZgCCaACtgbCAgoQABiwAxjWBBhHwgIHECMYsAIYJ8ICCBAAGIAEGKIEmAMAiAYBkAYIkgcDNS40oAe4ObIHAzQuNLgHrAY&sclient=gws-wiz-serp)

### Below is the recommended order for analyses.

### 1) Visual Data Analyses (Pre-training)
The Jupyter notebook [pre_plot_analyses.ipynb](pre_plot_analyses.ipynb) will output data statistics and the temporal and spatial distribution of variables.

### 2) Data Preprocessing
The script [preprocessing.py](preprocessing.py) scales the dataset, compute correlation and does a simple dataset split using a given time. </br>
To save the training and test datasets in your locally, remove comments from the last part of code.

### 3) Training and Evaluation
The script [train_and_evaluate.py](train_and_evaluate.py) has following parts:
- Function to load train-test data and determine the predictors(X) and predictands(y). 
The predictand/s could be chosen as clf, lwp or both. The predictors remain fixed as all other variables.
- Functions to train different models.
- Function to evaluate the chosen model performance. 
- Some other functions specific to models (eg, get_mlr_coefficients)
- shap_explainability model (needs troubleshooting)

### 4) Visual Data Analyses (Post-training)
The script [train_evaluate_analyses.ipynb](train_evaluate_analyses.ipynb) calls the functions from other python scripts to </br>
- Evaluate the results of training models and some visual plots

## Contributing to this repository

To contribute to this project, create a new branch using: </br>
`git checkout -b <branch_name>` </br>
Commit and push your edits to this new branch. And then create a pull request on Github.
