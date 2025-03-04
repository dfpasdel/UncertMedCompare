import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_bland_altman_example(weighting=None):
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt_a = np.random.uniform(20, 50, 10000)
    std_x_a = 5
    std_y_a = 8
    x_gt_b = np.random.uniform(50, 80, 10000)
    std_x_b = 5
    std_y_b = 2
    pbias = 1.2

    x_a = x_gt_a + np.random.normal(0, std_x_a, len(x_gt_a))
    x_b = x_gt_b + np.random.normal(0, std_x_b, len(x_gt_b))
    y_a = x_gt_a * pbias + np.random.normal(0, std_y_a, len(x_gt_a))
    y_b = x_gt_b * pbias + np.random.normal(0, std_y_b, len(x_gt_b))

    # Original samples
    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])

    # Alternate samples with different distribution of the mean values
    idx_low = 0.5 * (x + y) <= 50
    idx_high = 0.5 * (x + y) > 50
    low_x, low_y = x[idx_low], y[idx_low]
    high_x, high_y = x[idx_high], y[idx_high]
    x_undersampled = np.concatenate([low_x[:1000], high_x])
    y_undersampled = np.concatenate([low_y[:1000], high_y])

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      range_of_interest=[30, 70],
                                      binwidth=1,
                                      reference_method_type="soft",
                                      weighting=weighting,
                                      confidence_interval=None,
                                      bootstrap_samples=1000)
    weighted_comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                            new_method_measurements=y_undersampled,
                                                            range_of_interest=[30, 70],
                                                            binwidth=1,
                                                            reference_method_type="soft",
                                                            weighting=weighting,
                                                            confidence_interval=None,
                                                            bootstrap_samples=1000)

    #############################################
    #                   Plot                    #
    #############################################
    xlim = [0, 100]
    ylim = [-40, 40]
    comparator.set_style()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharex="col")
    comparator.plot_bland_altman(xlim=xlim, ylim=ylim, ax=axs[0, 0],
                                 provide_correlation=True, title="Original sampling")
    weighted_comparator_undersampled.plot_bland_altman(xlim=xlim, ylim=ylim, ax=axs[0, 1],
                                                       provide_correlation=True, title="Under sampling")
    sns.histplot(x=0.5 * (x + y), binwidth=1, ax=axs[1, 0])
    axs[1, 0].set_xlim(xlim)
    sns.histplot(x=0.5 * (x_undersampled + y_undersampled), binwidth=1, ax=axs[1, 1])
    axs[1, 1].set_xlim(xlim)
    plt.suptitle(str(weighting) + " weighting")
    plt.tight_layout()
    plt.show()
    comparator.reset_style()


if __name__ == '__main__':
    # No weigthing: mean and LoAs are different
    run_bland_altman_example(weighting=None)

    # Inverse weighting: mean and LoAs are similar
    run_bland_altman_example(weighting="inverse")
