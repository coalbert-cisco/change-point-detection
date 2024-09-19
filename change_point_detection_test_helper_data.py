import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from change_point_detection_helper import threshold_finder

# Set a random seed for reproducibility
np.random.seed(42)

# Testing Parameters
number_of_blocks = 5
sampling_block_size = 20
alarm_threshold = threshold_finder(
    desired_average_run_length=5000, sampling_block_size=sampling_block_size
)

# Testing Data. Kernel is set to median pairwise distances between data.
# Sudden Mean Shift
data1 = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
sudden_mean_shift_data = pd.Series(data1, name="Sudden Mean Shift")
sudden_mean_shift_kernel_bandwidth = np.median(pdist(sudden_mean_shift_data))

# Gradual Mean Shift
data2 = np.concatenate(
    [np.random.normal(0, 1, 100), np.random.normal(0.1 * np.arange(50), 1)]
)
gradual_mean_shift_data = pd.Series(data2, name="Gradual Mean Shift")
gradual_mean_shift_kernel_bandwidth = np.median(pdist(gradual_mean_shift_data))

# Sudden Variance Shift
data3 = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(0, 5, 100)])
sudden_variance_shift_data = pd.Series(data3, name="Sudden Variance Shift")
sudden_variance_shift_kernel_bandwidth = np.median(pdist(sudden_variance_shift_data))

# Periodic Signal With Noise
data4 = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.5, 1000)
periodic_signal_with_noise_data = pd.Series(data4, name="Periodic Signal with Noise")
periodic_signal_with_noise_kernel_bandwidth = np.median(
    pdist(periodic_signal_with_noise_data)
)

# Random Walk With Drift
data5 = np.cumsum(np.random.normal(0, 1, 1000)) + np.linspace(0, 10, 1000)
random_walk_with_drift_data = pd.Series(data5, name="Random Walk with Drift")
random_walk_with_drift_kernel_bandwidth = np.median(pdist(random_walk_with_drift_data))
