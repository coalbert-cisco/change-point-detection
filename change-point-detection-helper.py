import pandas as pd
import numpy as np

from scipy.stats import normal
from scipy.optimize import brentq

# Helper functions for Scan B-Statistic with Kernel (SBSK) algorithm.
# Initially reproducing functions from this repo: https://github.com/Wang-ZH-Stat/SBSK/tree/main
# I will be optimizing these moving forward.


# Reproducing Hilbert Space kernel.
# Mathematical background: https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space
def reproducing_hilbert_space_kernel(
    x: np.array, y: np.array, kernel_bandwidth: float, kernel: str = "Gaussian"
) -> float:
    # DOCUSTRING GOES HERE
    if kernel == "Gaussian":
        return np.exp(-np.sum((x - y) ** 2) / (2 * kernel_bandwidth**2))
    elif kernel == "Laplacian":
        return np.exp(-np.sum(np.abs(x - y)) / kernel_bandwidth)


# Used for calculating maximum mean discrepency (MMD).
def u_statistic_kernel(
    x1: np.array,
    y1: np.array,
    x2: np.array,
    y2: np.array,
    kernel_bandwidth: float,
    kernel: str = "Gaussian",
) -> float:
    # DOCUSTRING GOES HERE
    return (
        reproducing_hilbert_space_kernel(
            x=x1, y=x2, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
        + reproducing_hilbert_space_kernel(
            x=y1, y=y2, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
        - reproducing_hilbert_space_kernel(
            x=x1, y=y2, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
        - reproducing_hilbert_space_kernel(
            x=x2, y=y1, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
    )


# Unbiased estimator used for B-test.
def maximum_mean_discrepency_squared(
    X: np.ndarray, Y: np.ndarray, kernel_bandwidth: float, kernel: str = "Gaussian"
) -> float:
    # DOCUSTRING GOES HERE
    n = X.shape[0]

    u_statistic_sum = sum(
        u_statistic_kernel(
            X[i, :],
            Y[i, :],
            X[j, :],
            Y[j, :],
            kernel_bandwidth=kernel_bandwidth,
            kernel=kernel,
        )
        for i in range(n)
        for j in range(n)
        if i != j
    )

    return u_statistic_sum / (n * (n - 1))


# Special function used for calculating average run length for online statistic of B-statistic.
def average_run_length_normal_helper(mu: float) -> float:
    # DOCUSTRING GOES HERE
    pdf_value = normal.pdf(mu / 2)
    cdf_value = normal.cdf(mu / 2)

    return (2 / mu) * (cdf_value - 0.5) / (mu / 2 * cdf_value + pdf_value)


# Calculate average run length.
def average_run_length(sampling_block_size: int, alarm_threshold: float) -> float:
    # DOCUSTRING GOES HERE
    inner_term_first = (2 * sampling_block_size - 1) / (
        np.sqrt(2 * np.pi) * sampling_block_size * (sampling_block_size - 1)
    )
    temp_mu = alarm_threshold * np.sqrt(
        2
        * (2 * sampling_block_size - 1)
        / (sampling_block_size * (sampling_block_size - 1))
    )
    helper_value = average_run_length_normal_helper(temp_mu)

    return (
        np.exp(alarm_threshold**2)
        / (2 * alarm_threshold)
        / (inner_term_first * helper_value)
    )


# THIS HAS CONSTANTS THAT NEED TO BE ADDRESSED
# Find desired threshold.
def threshold_finder(
    desired_average_run_length: float, sampling_block_size: int
) -> float:
    # DOCUSTRING GOES HERE
    def threshold_helper(alarm_threshold: float) -> float:
        return desired_average_run_length - average_run_length(
            sampling_block_size=sampling_block_size, alarm_threshold=alarm_threshold
        )

    return brentq(threshold_helper, a=2, b=5, xtol=1e-4)


def b_statistic_variance(
    X: np.ndarray,
    number_of_blocks: int,
    sampling_block_size: int,
    kernel_bandwidth: float,
    iterations: int = 10000,
    kernel: str = "Gaussian",
    improve: bool = False,
) -> float:
    # DOCUSTRING GOES HERE
    n = X.shape[0]
    variance_sum = 0
    probability_list = np.ones(n)

    for _ in range(iterations):
        if not improve:
            id = np.random.choice(
                np.arange(1, n + 1), 6, replace=False
            )  # 6 because each u-statistic takes four inputs, but two are reused. This was an error in the R implementation
        else:
            id = np.random.choice(
                np.arange(1, n + 1), 6, replace=False, p=probability_list
            )
            probability_list[id] -= 1 / iterations
            probability_list /= np.sum(probability_list)

        sample_row = X[id, :]
        sample_u_stat_1 = u_statistic_kernel(
            x1=sample_row[0, :],
            y1=sample_row[1, :],
            x2=sample_row[2, :],
            y2=sample_row[3, :],
            kernel_bandwidth=kernel_bandwidth,
            kernel=kernel,
        )
        sample_u_stat_2 = u_statistic_kernel(
            x1=sample_row[4, :],
            y1=sample_row[1, :],
            x2=sample_row[5, :],
            y2=sample_row[3, :],
            kernel_bandwidth=kernel_bandwidth,
            kernel=kernel,
        )
        variance_sum += (
            sample_u_stat_1**2 + sample_u_stat_2**2
        ) / (  # the division of 2N instead of N is not explained
            2 * number_of_blocks
        ) + sample_u_stat_1 * sample_u_stat_2 * (
            number_of_blocks - 1
        ) / number_of_blocks

    variance_sum *= (
        1 / iterations * 2 / (sampling_block_size * (sampling_block_size - 1))
    )
    return variance_sum


def detect_change_points(
    X: np.ndarray,
    number_of_blocks: int,
    sampling_block_size: int,
    alarm_threshold: float,
    kernel_bandwidth: float,
    kernel: str = "Gaussian",
    improve: bool = False,
) -> tuple[int, int]:
    # DOCUSTRING GOES HERE
    n = X.shape[0]

    process_variance = b_statistic_variance(
        X=X[: (n // 2), :],
        number_of_blocks=number_of_blocks,
        sampling_block_size=sampling_block_size,
        kernel_bandwidth=kernel_bandwidth,
        kernel=kernel,
        improve=improve,
    )

    for time in range(n // 2, n + 1):
        temp_B_statistic = 0
        Y_s = X[(time - sampling_block_size) : time, :]  # Sample time block size

        for _ in range(number_of_blocks):
            sampled_indices = np.random.choice(
                time - sampling_block_size, sampling_block_size, replace=False
            )
            X_s = X[sampled_indices, :]  # Sampled submatrix

            temp_B_statistic += (
                maximum_mean_discrepency_squared(
                    X=X_s, Y=Y_s, kernel_bandwidth=kernel_bandwidth, kernel=kernel
                )
                / number_of_blocks
            )

        # The online scan B-statistic exceeds the threshold
        if temp_B_statistic / np.sqrt(process_variance) > alarm_threshold:
            return (time, 1)

        if time == n:
            # fail to detect
            return (n, 0)


# The rest of the functions in the source repo are functions for comparison.
# It was shown that the B-statistic was superior so I will leave the implementation at this.
# Something something
