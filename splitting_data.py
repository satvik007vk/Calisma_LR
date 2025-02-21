# Ensure time is sorted
ds_scaled = ds_scaled.sortby('time')

# Get time values
time_values = ds_scaled['time'].values

# Compute split index (80% train, 20% test)
split_index = int(len(time_values) * 0.8)
train_time = time_values[:split_index]
test_time = time_values[split_index:]

# Split dataset
ds_train = ds_scaled.sel(time=train_time)
ds_test = ds_scaled.sel(time=test_time)

# Print split summary
print(f"Training period: {train_time[0]} to {train_time[-1]}")
print(f"Testing period: {test_time[0]} to {test_time[-1]}")

print(f"Train size: {ds_train.sizes}")
print(f"Test size: {ds_test.sizes}")
