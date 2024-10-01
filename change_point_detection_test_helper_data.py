import numpy as np
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

time_steps = 100
sample_size = 100

# Testing Data. Kernel is set to median pairwise distances between data.
# Sudden Mean Shift
data1_temp1, data1_temp2 = (
    np.random.multivariate_normal(
        mean=np.zeros(time_steps), cov=np.eye(time_steps), size=sample_size
    ),
    np.random.multivariate_normal(
        mean=np.ones(time_steps), cov=np.eye(time_steps), size=sample_size
    ),
)

sudden_mean_shift_data = np.vstack((data1_temp1, data1_temp2))
sudden_mean_shift_kernel_bandwidth = np.median(pdist(sudden_mean_shift_data))

# Gradual Mean Shift
data2_temp1, data2_temp2 = (
    np.random.multivariate_normal(
        mean=np.zeros(time_steps), cov=np.eye(time_steps), size=sample_size
    ),
    np.random.multivariate_normal(
        mean=0.1 * np.arange(time_steps), cov=np.eye(time_steps), size=sample_size
    ),
)

gradual_mean_shift_data = np.vstack((data2_temp1, data2_temp2))
gradual_mean_shift_kernel_bandwidth = np.median(pdist(gradual_mean_shift_data))

# Sudden Variance Shift
data3_temp1, data3_temp2 = (
    np.random.multivariate_normal(
        mean=np.zeros(time_steps), cov=np.eye(time_steps), size=sample_size
    ),
    np.random.multivariate_normal(
        mean=np.zeros(time_steps), cov=np.eye(time_steps) * 5, size=sample_size
    ),
)

sudden_variance_shift_data = np.vstack((data3_temp1, data3_temp2))
sudden_variance_shift_kernel_bandwidth = np.median(pdist(sudden_variance_shift_data))

# Periodic Signal With Noise
periodic_signal_with_noise_data = np.random.multivariate_normal(
    mean=np.sin(np.linspace(0, 10 * np.pi, time_steps)),
    cov=np.eye(time_steps) / 2,
    size=sample_size,
)
periodic_signal_with_noise_kernel_bandwidth = np.median(
    pdist(periodic_signal_with_noise_data)
)

# Gaussian to Laplace
data5_temp1, data5_temp2 = (
    np.random.normal(size=(time_steps, 1)),
    np.random.laplace(size=(100, 1)),
)
gaussian_to_laplace_data = np.vstack((data5_temp1, data5_temp2))
gaussian_to_laplace_kernel_bandwidth = np.median(pdist(gaussian_to_laplace_data))
