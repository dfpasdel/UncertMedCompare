import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_range_of_interest_example():
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt = np.random.uniform(10, 90, 10000)
    std_x = 0.
    std_y = 2.

    # Simulate a non-linear model
    # NOTE: An AI algorithm that intends to regress a value does not necessarily have a linear behavior
    def f(x):
        return ((x / 100 - 0.2) ** 2 + 0.16) * 150 + np.random.normal(0, std_y, len(x))

    x = x_gt + np.random.normal(0, std_x, len(x_gt))
    y = f(x_gt)

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      range_of_interest=[20, 45],
                                      binwidth=1,
                                      reference_method_type="hard",
                                      weighting=None)
    comparator_bis = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[55, 80],
                                          binwidth=1,
                                          reference_method_type="hard",
                                          weighting=None)

    #############################################
    #                   Plot                    #
    #############################################
    xlim = [0, 100]
    ylim = [0, 100]
    comparator.set_style()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 7), sharex="col")
    comparator.plot_regression(xlim=xlim, ylim=ylim, ax=axs[0], title="Original sampling")
    comparator_bis.plot_regression(xlim=xlim, ylim=ylim, ax=axs[1], title="Under sampling")
    plt.suptitle("Range of interest example")
    plt.tight_layout()
    plt.show()
    comparator.reset_style()


if __name__ == '__main__':
    run_range_of_interest_example()
