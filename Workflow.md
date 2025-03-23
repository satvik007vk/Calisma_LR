
## Calisma Lab Rotation Workflow

### Visualisation

- [x] The CSV was converted to Xarray with dimensions as time, latitude, and longitude.
- [x] Analyse temporal distribution by taking a mean (or a median) across latitudes and longitudes.
- [x] Analyse spatial distribution by taking a mean (or a median) across all time.

Questions:
- Shall visual analyses be used to make any decisions in data preparation or is it just an overview of Data?

### Preparing Dataset

- [x] Scale the Variables using Standard/Min-Max Scaling.
- [ ] #TODO - remove outliers
- [x] Split the variables into train-test by a time to avoid overlap of both the sets. 
- [x] Feed them into the ML models multiple linear regression, XG boost, random forest.

Questions:
- should the predictands ('lwp' and 'clf') be scaled as well?
- What is the effort value of trying to do a better split (Rolling window split, timeseries split etc.)-
https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088
- There are only a few correlated variables (rh850-q850 (0.9),tcwv-q700,q850,rh700 (0.8)- but these are not handled now in preprocessing. Should some preprocessing involve changing these?

### Training and Evaluation
- [ ] Use models mlr, xgboost and random forest so far
- [ ] Train with clf and lwp separately, then both together.
- [ ] RSME and MSE calculated.
- [ ] Calculate contribution of each variable using mlr coefficients

Questions:
- Any other evaluation/ analyses to calculate?
- Any other model suggestion?
