import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from UncertMedCompare.continuous_comparator import ContinuousComparator


def heteroscedasticity_check_and_plot(reference_method_measurements, new_method_measurements, reference_method_type,
                                      title=""):
    comparator = ContinuousComparator(reference_method_measurements=reference_method_measurements,
                                      new_method_measurements=new_method_measurements,
                                      reference_method_type=reference_method_type)
    comparator.set_style()
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    # heteroscedasticity_info = comparator.heteroscedasticity_info
    if reference_method_type == "hard":
        comparator.plot_regression(ax=axs[0])
    else:
        comparator.plot_bland_altman(xlim=[0, 1], ylim=[-0.5, 0.5], ax=axs[0])
    comparator.plot_heteroscedasticity(ax=axs[1])
    fig.suptitle(title)
    fig.show()
    plt.close(fig)
    comparator.reset_style()


def no_heteroscedasticity_hard_label_example():
    x_gt = np.random.normal(0.5, 0.15, 50000)
    x = x_gt
    y = x_gt + np.random.normal(0, 0.01, len(x_gt))
    heteroscedasticity_check_and_plot(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="hard",
                                      title="No heteroscedasticity - Hard label")


def monotonous_heteroscedasticity_hard_label_example():
    x_gt = np.random.normal(0.5, 0.15, 50000)
    x = x_gt
    y = []
    for i in range(len(x_gt)):
        y.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * x_gt[i])))
    y = np.array(y)
    heteroscedasticity_check_and_plot(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="hard",
                                      title="Monotonous heteroscedasticity - Hard label")


def non_monotonous_heteroscedasticity_hard_label_example():
    x_gt = np.random.normal(0.5, 0.15, 50000)
    x = x_gt
    y = []
    for i in range(len(x_gt)):
        y.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * (0.5 - abs(0.5 - x_gt[i])))))
    y = np.array(y)
    heteroscedasticity_check_and_plot(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="hard",
                                      title="Non-monotonous heteroscedasticity - Hard label")


def no_heteroscedasticity_soft_label_example():
    # In this example, it is illustrated that heteroscedasticity can also be caused by uncertainty in the reference
    # method (i.e. soft values) without the error being dependent on the magnitude of the measurements.
    # This underlines the necessity of qualitative assessment of heteroscedasticity as well
    x_gt = np.random.uniform(0.45, 0.55, 50000)
    x = x_gt + np.random.normal(0, 0.1, len(x_gt))
    y = x_gt + np.random.normal(0, 0.1, len(x_gt))
    heteroscedasticity_check_and_plot(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="soft",
                                      title="No heteroscedasticity - Soft label\nHeteroscedasticity detected because "
                                            "of soft label")


def run_heteroscedasticity_detection_example():
    no_heteroscedasticity_hard_label_example()
    monotonous_heteroscedasticity_hard_label_example()
    non_monotonous_heteroscedasticity_hard_label_example()
    no_heteroscedasticity_soft_label_example()


if __name__ == '__main__':
    run_heteroscedasticity_detection_example()
