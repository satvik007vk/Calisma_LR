
## Lab rotation all steps

### Visualisation

- [x] The CSV was converted to Xarray with dimensions as time, latitude, and longitude.
- [x] Analyse temporal and spatial distribution of Variables
- [x] The cloud fraction and liquid water path were plotted against time

Questions:
- Shall visual analyses help in any sort of data preparation or is it just an overview of Data?

### Preparing Dataset
- [x] Scale the Variables using Standard/Min-Max Scaling. 
- [x] Split the variables into train-test by a time to avoid overlap of both the sets. 
- [x] Feed them into the ML models multiple linear regression, XG boost, random forest.
- [ ] First predict clf separately, then lwp and then both together?

- What should change in this?