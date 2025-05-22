
## Results
#TODO- Get results using Min Max scaler

### Below results using Sklearn libraries (do not support combined loss of multiple targets)

### Using a standard scalar and Outliers Removed using IQR.

The numbers represent Root Mean Square Error (RMSE): Higher values correspond to worse performance.

| Model        | Predictand LWP | Predictand CLF | Predictand (LWP + CLF) |
|--------------|----------------|----------------|------------------------|
| MLR          | 0.83           | 0.24           | 0.54                   |
| XGBoost      | 0.78           | 0.22           | 0.50                   |
| Random Forest|           |                |                        |

#### Contributions for MLR (when both are predictands)
Top 5 for contributors 'clf':
rh700    0.194104
eis      0.182463
rh850    0.174769
t850     0.160325
lnNd     0.128220

Top 5 for contributors 'lwp':
eis      0.434696
tcwv     0.423618
rh850    0.306584
t850     0.270461
blh      0.260107


## Poor Results

### Using a standard scalar and Outliers Removed using percentile (3 and 97).

The numbers represent Root Mean Square Error (RMSE): Higher values correspond to worse performance.

| Model        | Predictand LWP | Predictand CLF | Predictand (LWP + CLF) |
|--------------|----------------|----------------|------------------------|
| MLR          | 0.84           | 0.80           | 0.82                   |
| XGBoost      | 0.79           | 0.74           | 0.77                   |
| Random Forest|                |                |                        |



### Using a standard scalar and No Outlier Removal.

The numbers represent Root Mean Square Error (RMSE): Higher values correspond to worse performance.

| Model        | Predictand LWP | Predictand CLF | Predictand (LWP + CLF) |
|--------------|----------------|----------------|------------------------|
| MLR          | 0.81           | 0.78           | 0.80                   |
| XGBoost      | 0.76           | 0.74           | 0.75                   |
| Random Forest| 0.79           | 0.73           | 0.75                   |

Major contributors to lwp: </br> eis (0.45) ; tcwv (0.43) ; q700 (-0.40)



