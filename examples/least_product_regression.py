import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_least_product_regression_example(weighting=None):
    #############################################
    #           Simulate some data              #
    #############################################
    mean_a = 30
    x_gt_a = np.random.normal(mean_a, 7, 100000)
    mean_b = 70
    x_gt_b = np.random.normal(mean_b, 7, 100000)
    std_x = 2.
    std_y = 2.

    # Simulate a non-linear model
    # NOTE: An AI algorithm that intends to regress a value does not necessarily have a linear behavior
    def f(x):
        return ((x / 100 - 0.2) ** 2 + 0.16) * 150 + np.random.normal(0, std_y, len(x))

    x_a = x_gt_a + np.random.normal(0, std_x, len(x_gt_a))
    x_b = x_gt_b + np.random.normal(0, std_x, len(x_gt_b))
    y_a = f(x_gt_a)
    y_b = f(x_gt_b)

    # Original samples
    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])

    # Alternate samples with different distribution of the label values
    x_undersampled = np.concatenate([x_a[:2000], x_b])
    y_undersampled = np.concatenate([y_a[:2000], y_b])

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      range_of_interest=[20, 80],
                                      binwidth=1,
                                      reference_method_type="soft",
                                      weighting=weighting,
                                      soft_regression_method="LP")
    weighted_comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                            new_method_measurements=y_undersampled,
                                                            range_of_interest=[20, 80],
                                                            binwidth=1,
                                                            reference_method_type="soft",
                                                            weighting=weighting,
                                                            soft_regression_method="LP")

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
    # No weigthing: the regression lines are different
    run_least_product_regression_example(weighting=None)

    # Inverse weighting: the regression lines are similar
    run_least_product_regression_example(weighting="inverse")
