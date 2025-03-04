import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from UncertMedCompare.continuous_comparator import ContinuousComparator


def run_values_formatting_example():
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt = np.random.normal(50, 10, 100000)
    std_x = 0.
    std_y = 2.
    x = x_gt + np.random.normal(0, std_x, len(x_gt))
    y = x_gt + np.random.normal(0, std_y, len(x_gt))

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      binwidth=1,
                                      reference_method_type="hard",
                                      format_dimensional="{:.01f}",
                                      # Format for dimensional values (ex: MAE, SD, ...). One decimal chosen here
                                      format_non_dimensional="{:.08f}",
                                      # Format for non-dimensional values (ex: slope, R2). Eight decimals chosen here
                                      units=" Fill unit here",
                                      )

    #############################################
    #                   Plot                    #
    #############################################
    comparator.set_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    comparator.plot_regression(xlim=[0, 100],
                               ylim=[0, 100],
                               title="Fill plot title here",
                               ax=ax)
    plt.tight_layout()
    plt.show()
    comparator.reset_style()


def run_layout_example():
    #############################################
    #           Simulate some data              #
    #############################################
    x_gt = np.random.normal(50, 10, 100000)
    std_x = 0.
    std_y = 2.
    x = x_gt + np.random.normal(0, std_x, len(x_gt))
    y = x_gt + np.random.normal(0, std_y, len(x_gt))

    #############################################
    #              Run analysis                 #
    #############################################
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      binwidth=1,
                                      reference_method_type="soft"
                                      )

    #############################################
    #                   Plot                    #
    #############################################
    comparator.set_style()

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    comparator.plot_regression(xlim=[0, 100],
                               ylim=[0, 100],
                               ax=axs[0],
                               title="With full legend",
                               show_legend=True,
                               show_legend_title=True,
                               plot_linreg=True)
    comparator.plot_regression(xlim=[0, 100],
                               ylim=[0, 100],
                               ax=axs[1],
                               title="Without full legend",
                               show_legend=True,
                               show_legend_title=False,
                               plot_linreg=False)
    fig.suptitle("Regression layout example")
    fig.show()
    plt.close(fig)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    comparator.plot_bland_altman(xlim=[0, 100], ylim=[-20, 20], ax=axs[0], title="With full legend",
                                 show_legend=True,
                                 show_legend_title=True,
                                 provide_slope=True,
                                 provide_correlation=True)
    comparator.plot_bland_altman(xlim=[0, 100], ylim=[-20, 20], ax=axs[1], title="Without full legend",
                                 show_legend=True,
                                 show_legend_title=False,
                                 provide_slope=True,
                                 provide_correlation=False)
    fig.suptitle("Bland-Altman layout example")
    fig.show()
    plt.close(fig)

    comparator.reset_style()


if __name__ == '__main__':
    run_values_formatting_example()
    run_layout_example()
