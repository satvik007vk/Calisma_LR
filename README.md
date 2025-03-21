# Calisma

Welcome to the Calisma Project repository. 
Below are the instructions to run the analyses


First, create the conda environment and install dependencies by running the following script:
`conda env create -f environment.yml` 

Then, activate your environment using:
`conda activate calisma`
Select this [conda environment as your python interpreter.](https://www.google.com/search?q=select+existing+conda+environment+as+python+interpreter&client=ubuntu-sn&hs=rkJ&sca_esv=63c9e37f8da915f5&channel=fs&sxsrf=AHTn8zqxaLTBEhH0yx9c5RTSDwiCU5Jqzw%3A1742553270476&ei=tkDdZ87cHIWui-gP_4WP6QE&ved=0ahUKEwjOj8Hl_JqMAxUF1wIHHf_CIx0Q4dUDCBA&uact=5&oq=select+existing+conda+environment+as+python+interpreter&gs_lp=Egxnd3Mtd2l6LXNlcnAiN3NlbGVjdCBleGlzdGluZyBjb25kYSBlbnZpcm9ubWVudCBhcyBweXRob24gaW50ZXJwcmV0ZXIyBRAAGO8FMgUQABjvBTIFEAAY7wUyBRAAGO8FMggQABiiBBiJBUiHGVCzBVipEnABeAGQAQCYAYYBoAGiBqoBAzYuM7gBA8gBAPgBAZgCCaACtgbCAgoQABiwAxjWBBhHwgIHECMYsAIYJ8ICCBAAGIAEGKIEmAMAiAYBkAYIkgcDNS40oAe4ObIHAzQuNLgHrAY&sclient=gws-wiz-serp)

### 1) Visual Data Analyses (Pre-training)
Run the Jupyter notebook [pre_plot_analyses.ipynb](pre_plot_analyses.ipynb) to visualise the temporal and spatial distribution of variables.
And some data statistics.

### 2) Data Preprocessing
The script [preprocessing.py](preprocessing.py) scales the dataset, compute correlation and does a simple dataset split using a given time.
In order to save the training and test data as CSVs. To save the datasets in your locally, remove comments from the last part of code.

### 3) Training and Evaluation
The script [train_and_evaluate.py](train_and_evaluate.py) has following parts:
- The function to load train-test data and determine the predictors(X) and predictands(y). 
The predictand could be choosen as clf, lwp or both. The preictors remain fixed as all other variables.
- The functions to train different models.
- The function to evaluate the chosen model performance. 
- Some other functions specific to models (eg, get_mlr_coefficients)
- shap_explainability model

### 4) Visual Data Analyses (Post-training)
- some functions to visualise the results of training models (eg: plot to see variable contribution as determined by mlr)