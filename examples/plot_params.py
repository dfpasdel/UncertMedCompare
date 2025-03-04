import matplotlib.pyplot as plt
import numpy as np

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_custom_params_example():
    """
    Run the custom plot parameters example
    The example demonstrates how to set custom parameters for the bland-altman plot
    It first simulates sample data and then runs the ContinuousComparator on the data
    Next, it plots two the bland-altman plot two times:
    1. With default parameters
    2. With custom parameters
    """
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt = np.random.normal(50, 10, 10000)
    std_x = 2.
    std_y = 2.
    x = x_gt + np.random.normal(0, std_x, len(x_gt))
    y = x_gt + np.random.normal(0, std_y, len(x_gt))

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="soft")
    xlim = [0, 100]
    ylim = [-40, 40]

    #############################################
    #                   Plot                    #
    #############################################
    comparator.set_style()
    # default parameters
    comparator.plot_bland_altman(xlim=xlim, ylim=ylim)
    plt.show()

    # custom parameters
    comparator.plot_bland_altman(xlim=xlim, ylim=ylim,
                                 title_fontsize=22,
                                 xticks_fontsize=12,
                                 yticks_fontsize=12,
                                 label_size=16,
                                 text_fontsize=12)
    plt.show()
    comparator.reset_style()


if __name__ == '__main__':
    run_custom_params_example()
