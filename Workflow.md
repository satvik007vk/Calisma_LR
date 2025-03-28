
## Calisma Lab Rotation Workflow

### Visualisation

- [x] Compute data statistics
- [x] Analyze temporal distribution using the mean or median across latitudes and longitudes.  
- [x] Analyze spatial distribution using the mean or median across time.
- [ ] Time series decomposition

### Preparing Dataset

- [x] Scale variables using Standard/Min-Max Scaling.  
- [x] Remove outliers using the interquartile range or fixed percentile.  
- [x] Split data into train-test sets based on a fixed time to prevent overlap.
- [ ] Drop correlated variables/ dimensionality reduction
- [x] Train ML models: Multiple Linear Regression, XGBoost, and Random Forest.

### Training and Evaluation

- [x] Train separately with CLF and LWP, then with both combined.  
- [x] Calculate RMSE and MSE for the models.  
- [x] Determine each variable's contribution using MLR coefficients.  
- [ ] Shap Explainability
- 

Questions/Potential next steps

- What preprocessing (or even later) decisions can be made using visual analyses? (currently its just a data overview)
- What is the effort value of trying to do a better split for time-series data (Rolling window split, timeseries split etc.)-
https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088
- Handling correlated variables: drop some variables, PCA etc.
  - There are only a few highly correlated variables (rh850-q850 (0.9),tcwv-q700,q850,rh700 (0.8)- but these are not handled now in preprocessing.
- Any other model suggestion?
- Any other helpful analyses?

