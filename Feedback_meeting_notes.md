
### Visualisation

Feedback HA:
for TS decomposition, groupby monthly average and  get seasonality and anomalies that are averaged out for years for one month. 
subtract seasonality from the dataset- to get trend and residuals


### Preparing Dataset

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

Feedback HA:
Check what happens and if we take both variables at a time, is it minimising the combined loss. In that case, predictands need to be scaled. because lwp value will otherwise overpower clf value.
Try a Neural Network.

Nd and clf, and Nd and lwp relationship.