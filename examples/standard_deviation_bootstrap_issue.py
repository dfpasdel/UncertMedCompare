import numpy as np
import matplotlib.pyplot as plt

from UncertMedCompare.utilities.misc import weighted_error_mean_and_std
from UncertMedCompare.continuous_comparator import ContinuousComparator


def illustrate_standard_deviation_bootstrapping_issue():
    """
    This example illustrates that bootstrapping standard deviations (and further the limts of agreement in a
    Bland-Altman analysis) leads to incorrect confidence intervals, that are not centered on the actual point estimate.
    Instead, the CIs are pulled towards 0.
    This is a motivation for not bootstrapping SDs and LOAs in the package.
    """
    sizes = [5 * i for i in range(2, 20)]
    average_distances_to_upper_quantile = []
    average_distances_to_lower_quantile = []
    for size in sizes:
        distances_to_upper_quantile = []
        distances_to_lower_quantile = []
        for _ in range(100):
            x_gt = np.random.normal(50, 10, size)
            std_x = 0.
            std_y = 2.
            x = x_gt + np.random.normal(0, std_x, len(x_gt))
            y = x_gt + np.random.normal(0, std_y, len(x_gt))

            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="hard",
                                              confidence_interval=95,
                                              bootstrap_samples=1000)
            [_, _, _], [point_estimate, lower_quantile, upper_quantile] = \
                comparator.bootstrap_metric(ref_values=x,
                                            new_values=y,
                                            func=weighted_error_mean_and_std)
            distances_to_upper_quantile.append(upper_quantile - point_estimate)
            distances_to_lower_quantile.append(lower_quantile - point_estimate)
        average_distances_to_upper_quantile.append(np.mean(distances_to_upper_quantile))
        average_distances_to_lower_quantile.append(np.mean(distances_to_lower_quantile))
    average_upper_quantiles = np.array(average_distances_to_upper_quantile) + std_y
    average_lower_quantiles = np.array(average_distances_to_lower_quantile) + std_y
    average_ci_centers = 0.5 * (average_lower_quantiles + average_upper_quantiles)
    fig, ax = plt.subplots()
    ax.axhline(y=std_y, color="gray", label="Point estimate SD")
    ax.plot(sizes, average_upper_quantiles, color="black", ls="", marker="_",
            label="95% CIs")
    ax.plot(sizes, average_lower_quantiles, color="black", ls="", marker="_")
    ax.plot(sizes, average_ci_centers, color="black", ls="", marker="o", label="CI center")
    ax.set_ylim([1, 3])
    ax.set_title("Standard deviation CIs pulled towards 0 when bootstrapping.\nCI center should be equal to point "
                 "estimate")
    ax.legend()
    fig.show()
    plt.close(fig)


if __name__ == '__main__':
    illustrate_standard_deviation_bootstrapping_issue()
