import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_regression_metrics_example(weighting=None):
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt_a = np.random.uniform(low=10, high=50, size=10000)
    x_gt_b = np.random.uniform(low=50, high=90, size=10000)

    # Simulate a linear model with heteroscedacity
    def f(x):
        return 0.5 * x + 35 + np.random.normal(0, 1, len(x)) * (0.6 + abs(x - 70) / 7)

    x_a = x_gt_a
    x_b = x_gt_b
    y_a = f(x_gt_a)
    y_b = f(x_gt_b)

    # Original samples
    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])

    # Alternate samples with different distribution of the label values
    x_undersampled = np.concatenate([x_a[:1000], x_b])
    y_undersampled = np.concatenate([y_a[:1000], y_b])

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x, new_method_measurements=y,
                                      binwidth=1,
                                      reference_method_type="hard", weighting=weighting)
    weighted_comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                            new_method_measurements=y_undersampled,
                                                            binwidth=1,
                                                            reference_method_type="hard", weighting=weighting)

    #############################################
    #                   Plot                    #
    #############################################
    xlim = [0, 100]
    ylim = [0, 100]
    comparator.set_style()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex="col")
    comparator.plot_regression(xlim=xlim, ylim=ylim, ax=axs[0, 0], title="Original sampling")
    weighted_comparator_undersampled.plot_regression(xlim=xlim, ylim=ylim, ax=axs[0, 1], title="Under sampling")
    sns.histplot(x=x, binwidth=1, ax=axs[1, 0])
    axs[1, 0].set_xlim(xlim)
    sns.histplot(x=x_undersampled, binwidth=1, ax=axs[1, 1])
    axs[1, 1].set_xlim(xlim)
    plt.suptitle(str(weighting) + " weighting")
    plt.tight_layout()
    plt.show()
    comparator.reset_style()


if __name__ == '__main__':
    # No weigthing: mean and LoAs are different
    run_regression_metrics_example(weighting=None)

    # Inverse weighting: mean and LoAs are similar
    run_regression_metrics_example(weighting="inverse")
