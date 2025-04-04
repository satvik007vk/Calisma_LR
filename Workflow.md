
## Calisma Lab Rotation Workflow

### Visualisation

- [x] Compute data statistics
- [x] Analyze temporal distribution using the mean or median across latitudes and longitudes.  
- [x] Analyze spatial distribution using the mean or median across time.
- [ ] Time series decomposition

Feedback HA:
for TS decomposition, groupby monthly average and  get seasonality and anomalies that are averaged out for years for one month. 
subtract seasonality from the dataset- to get trend and residuals

### Preparing Dataset

- [x] Scale variables using Standard/Min-Max Scaling.  
- [x] Remove outliers using the interquartile range or fixed percentile.  
- [x] Split data into train-test sets based on a fixed time to prevent overlap.
- [ ] Drop correlated variables/ dimensionality reduction

Feedback HA:
Splitting: randomly split is not good enough for atm science dataset
Outliers: Remove outliers only for good reasons, in Satellites measurements there are uncertainties.
May improve model performance, but maybe not predict outlier values.

Scaling: downside is finding the real unit of change across space, can be mitigated if we convert the variables back.
Comparison across different variables is easier in scaling but difficult to compare across space.

Try without scaling the predictands.

Correlation: makes the model less explainable about the contributions of correlated variables. 
Number of variables here are not so much, so not important to reduce them now.

TRY: drop one correlated variable, and see the model performance.
Recursively select the least important features, and drop. 
Scikitlearn can do it - https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html

### Training and Evaluation

- [x] Train ML models: Multiple Linear Regression, XGBoost, and Random Forest.
- [x] Train separately with CLF and LWP, then with both combined.  
- [x] Calculate RMSE and MSE for the models.  
- [x] Determine each variable's contribution using MLR coefficients.  
- [ ] Shap Explainability
Feedback HA:
Check what happens and if we take both variables at a time, is it minimising the combined loss. In that case, predictands need to be scaled. because lwp value will otherwise overpower clf value.
Try a Neural Network.

Nd and clf, and Nd and lwp relationship.

Questions/Potential next steps

- What preprocessing (or even later) decisions can be made using visual analyses? (currently its just a data overview)
- What is the effort value of trying to do a better split for time-series data (Rolling window split, timeseries split etc.)-
https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088
- Handling correlated variables: drop some variables, PCA etc.
  - There are only a few highly correlated variables (rh850-q850 (0.9),tcwv-q700,q850,rh700 (0.8)- but these are not handled now in preprocessing.
- Any other model suggestion?
- Any other helpful analyses?

