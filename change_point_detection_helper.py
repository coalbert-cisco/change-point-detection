import numpy as np

from scipy.stats import norm
from scipy.optimize import brentq

# Helper functions for Scan B-Statistic with Kernel (SBSK) algorithm.
# Initially reproducing functions from this repo: https://github.com/Wang-ZH-Stat/SBSK/tree/main
# I will be optimizing these moving forward.


def reproducing_kernel_hilbert_space(
    x: np.array, y: np.array, kernel_bandwidth: float, kernel: str = "Gaussian"
) -> float:
    """
    Reproducing kernel Hilbert space (RKHS). See link below for mathematical explanation.
    https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space

    Args:
        x (np.array): n-dimensional vector.
        y (np.array): n-dimensional vector.
        kernel_bandwidth (float): Kernel width that controls degree of smoothness and influence of range.
        kernel (str, optional): Desired radial basis function. Defaults to "Gaussian".

    Returns:
        float: Inner product of x and y on the feature space.
    """
    if kernel == "Gaussian":
        return np.exp(-np.sum((x - y) ** 2) / (2 * kernel_bandwidth**2))
    elif kernel == "Laplacian":
        return np.exp(-np.sum(np.abs(x - y)) / kernel_bandwidth)


def u_statistic_kernel(
    x1: np.array,
    y1: np.array,
    x2: np.array,
    y2: np.array,
    kernel_bandwidth: float,
    kernel: str = "Gaussian",
) -> float:
    """
    Kernel for U-statistic defined by RKHSs.
    U-statistic helper function of unbiased estimator of MMD^2 = 1 / (n(n - 1))\sum_{i\neq j}^n h(x_i, y_i, x_j, y_j)
    MMD is maximum mean discrepency: https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution

    Args:
        x1 (np.array): n-dimensional vector.
        y1 (np.array): n-dimensional vector.
        x2 (np.array): n-dimensional vector.
        y2 (np.array): n-dimensional vector.
        kernel_bandwidth (float): Kernel width that controls degree of smoothness and influence of range.
        kernel (str, optional): Desired radial basis function. Defaults to "Gaussian".

    Returns:
        float: Unbiased estimate of MMD statistic.
    """
    return (
        reproducing_kernel_hilbert_space(
            x=x1, y=x2, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
        + reproducing_kernel_hilbert_space(
            x=y1, y=y2, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
        - reproducing_kernel_hilbert_space(
            x=x1, y=y2, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
        - reproducing_kernel_hilbert_space(
            x=x2, y=y1, kernel_bandwidth=kernel_bandwidth, kernel=kernel
        )
    )


def maximum_mean_discrepency_squared(
    X: np.ndarray, Y: np.ndarray, kernel_bandwidth: float, kernel: str = "Gaussian"
) -> float:
    """
    Unbiased estimate of MMD^2 via the above U-statistic.

    Args:
        X (np.ndarray): m by n iid domain sample of distribution P.
        Y (np.ndarray): l by n iid domain sample of distribution Q.
        kernel_bandwidth (float): Kernel width that controls degree of smoothness and influence of range.
        kernel (str, optional): Desired radial basis function. Defaults to "Gaussian".

    Returns:
        float: MMD^2, otherwise known as the B-test statistic.
    """
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


def average_run_length_normal_helper(mu: float) -> float:
    """
    Bespoke function as helper in calculating average run length (ARL).
    The use of this function was not explained in the literature beyond a helper function.
    There was no further academic explanation upon further search.

    Args:
        mu (float): Mean parameter for normal distribution.

    Returns:
        float: Calculated helper value for ARL.
    """
    pdf_value = norm.pdf(mu / 2)
    cdf_value = norm.cdf(mu / 2)

    return (2 / mu) * (cdf_value - 0.5) / (mu / 2 * cdf_value + pdf_value)


# Calculate average run length.
def average_run_length(sampling_block_size: int, alarm_threshold: float) -> float:
    """
    Calculates average run length in online scan of B-statistic where B_0 (sampling_block_size) \geq 2.
    This is technically the mean stopping time of T, the infimum of times that the online scan B-statistic is greater than the threshold.

    Args:
        sampling_block_size (int): Non-overlapping fixed block size (the most recent samples, refers to the size of the reference block for MMD calculation).
        alarm_threshold (float): Stopping time threshold.

    Returns:
        float: Average run length over sample block.
    """
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
    """
    Threshold calculator for online scan B-statistic stopping time.

    Args:
        desired_average_run_length (float): How much time the ARL should be calculated over.
        sampling_block_size (int): Non-overlapping fixed block size.

    Returns:
        float: B-statistic stopping time threshold.
    """

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
    """
    Variance of the online scan B-statistic.

    There is either an implementation error in the R version or a syntax error in the original paper.
    The variance in the paper is defined to use points x, y, x', y', x'', and x'''.
    The R implementation uses all unique samples to calculate the covariance term: Cov(h(x, y, x', y'), h(x'', y'', x''', y''')) instead of Cov(h(x, y, x', y'), h(x'', y, x''', y')).
    There was no explanation of this, so I will assume it was an implementation error.

    Also, the R implementation divides the second moment by 2N instead of N. There was no explanation for this.
    I believe this is an implementation error. If it is not, there is some symmetry in the second moment that I did not identify.
    The "2" will be removed until testing shows otherwise.


    Args:
        X (np.ndarray): m by n submatrix.
        number_of_blocks (int): Number of blocks.
        sampling_block_size (int): Non-overlapping fixed block size.
        kernel_bandwidth (float): Kernel width that controls degree of smoothness and influence of range.
        iterations (int, optional): Monte Carlo iterations. Defaults to 10000.
        kernel (str, optional): Desired radial basis function. Defaults to "Gaussian".
        improve (bool, optional): Whether to use an improved sampling sampling method that specifies sampling distribution. Defaults to False.

    Returns:
        float: Estimated variance of online scan B-statistic.
    """
    n = X.shape[0]
    variance_sum = 0
    probability_list = np.ones(n)

    for _ in range(iterations):
        if not improve:
            id = np.random.choice(
                np.arange(1, n + 1), 6, replace=False
            )  # 6 because each u-statistic takes four inputs, but two are reused
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
    """_summary_

    Args:
        X (np.ndarray): Full q by n data matrix.
        number_of_blocks (int): Number of blocks.
        sampling_block_size (int): Non-overlapping fixed block size.
        alarm_threshold (float): Stopping time threshold.
        kernel_bandwidth (float): Kernel width that controls degree of smoothness and influence of range.
        kernel (str, optional): Desired radial basis function. Defaults to "Gaussian".
        improve (bool, optional): Whether to use an improved sampling sampling method that specifies sampling distribution. Defaults to False.

    Returns:
        tuple[int, int]: (change point location, integer boolean whether change point was found)
    """
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
# This method is meant to be compared with the "ruptures" package, which is what we were using.
