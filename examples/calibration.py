import numpy as np
import matplotlib.pyplot as plt

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_calibration_hard_reference_example():
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt = np.random.normal(50, 5, 100000)
    std_x = 0.
    std_y = 2.
    pbias = 1.2
    fbias = 10
    x = x_gt + np.random.normal(0, std_x, len(x_gt))
    y = x_gt * pbias + fbias + np.random.normal(0, std_y, len(x_gt))

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      range_of_interest=[20, 80],
                                      binwidth=1,
                                      confidence_interval=False,
                                      reference_method_type="hard")
    calibration_function = comparator.calibration_function
    calibrated_y = calibration_function(y)
    comparator_calibrated = ContinuousComparator(reference_method_measurements=x,
                                                 new_method_measurements=calibrated_y,
                                                 range_of_interest=[20, 80],
                                                 binwidth=1,
                                                 confidence_interval=False,
                                                 reference_method_type="hard")

    #############################################
    #                   Plot                    #
    #############################################
    comparator.set_style()
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    comparator.plot_regression(xlim=[0, 100], ylim=[0, 100], ax=axs[0], title="Non calibrated")
    comparator_calibrated.plot_regression(xlim=[0, 100], ylim=[0, 100], ax=axs[1], title="Calibrated")
    plt.suptitle("Hard reference")
    plt.show()
    comparator.reset_style()


def run_calibration_soft_reference_example():
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt = np.random.uniform(20, 80, 100)
    std_x = 3.
    std_y = 3.
    pbias = 1.4
    fbias = 10
    x = x_gt + np.random.normal(0, std_x, len(x_gt))
    y = x_gt * pbias + fbias + np.random.normal(0, std_y, len(x_gt))

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      range_of_interest=[30, 70],
                                      binwidth=1,
                                      reference_method_type="soft")
    calibration_function = comparator.calibration_function
    calibrated_y = calibration_function(y)
    comparator_calibrated = ContinuousComparator(reference_method_measurements=x,
                                                 new_method_measurements=calibrated_y,
                                                 range_of_interest=[30, 70],
                                                 binwidth=1,
                                                 reference_method_type="soft")

    #############################################
    #                   Plot                    #
    #############################################
    comparator.set_style()
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    comparator.plot_bland_altman(xlim=[0, 100], ylim=[-40, 40], ax=axs[0],
                                 provide_correlation=True, title="Non calibrated")
    comparator_calibrated.plot_bland_altman(xlim=[0, 100], ylim=[-40, 40], ax=axs[1],
                                            provide_correlation=True, title="Calibrated")
    plt.suptitle("Soft reference")
    plt.show()
    comparator.reset_style()


if __name__ == '__main__':
    # Hard reference
    run_calibration_hard_reference_example()

    # Soft reference
    run_calibration_soft_reference_example()
