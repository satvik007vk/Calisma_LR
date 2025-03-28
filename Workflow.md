
## Calisma Lab Rotation Workflow

### Visualisation

- [x] Analyse temporal distribution by taking a mean (or a median) across latitudes and longitudes.
- [x] Analyse spatial distribution by taking a mean (or a median) across all time.



### Preparing Dataset

- [x] Scale the Variables using Standard/Min-Max Scaling.
- [x] Remove outliers using Inter quartile range/ fixed percentile
- [x] Split the variables into train-test by a time to avoid overlap of both the sets. 
- [x] Feed them into the ML models multiple linear regression, XG boost, random forest.

Questions:
- should the predictands ('lwp' and 'clf') be scaled as well?

### Training and Evaluation
- [x] Use models mlr, xgboost and random forest.
- [x] Train with clf and lwp separately, then both together.
- [x] RSME and MSE calculated.
- [ ] Calculate contribution of each variable using mlr coefficients

Questions/Possible change in analyses

- What preprocessing (or even later) decisions can be made using visual analyses? (currently its just a data overview)
- What is the effort value of trying to do a better split for time-series data (Rolling window split, timeseries split etc.)-
https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088
- Handling correlated variables: drop some variables, PCA etc.
  - There are only a few correlated variables (rh850-q850 (0.9),tcwv-q700,q850,rh700 (0.8)- but these are not handled now in preprocessing.
- Any other model suggestion?
- Any other helpful analyses?

