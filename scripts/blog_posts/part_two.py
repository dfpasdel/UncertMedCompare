import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr
from sklearn import metrics
from PIL import Image
import seaborn as sns

from UncertMedCompare import ContinuousComparator
from UncertMedCompare.config import DEFAULT_STYLE

# Plots params
plt.style.use(DEFAULT_STYLE)
alpha = 0.1
dpi_landscape = 100
dpi_portrait = 80
dpi_squared = 80

# Set seed for reproducibility
np.random.seed(0)

# Simulate some ground truth sample values
x_gt = np.random.uniform(5, 95, 2000)


####################################################
#  Evaluating a model with known reference values  #
####################################################

# Simulate a biased model M:
def M_biased(x):
    return x * 0.9 + 10 + np.random.normal(0, 2, len(x))

# Simulate age predictions from M
y = M_biased(x_gt)

# Initializing the comparator for "hard" reference values
comparator = ContinuousComparator(reference_method_measurements=x_gt,
                                  new_method_measurements=y,
                                  reference_method_type="hard")

# Generate scatter plot and compute standard metrics
fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi_squared)
comparator.plot_regression(xlim=[0, 100],
                           ylim=[0, 100],
                           xlabel="x",
                           ylabel="y = M(I)",
                           alpha=alpha,
                           ax=ax)
fig.tight_layout()
savepath = "./figures/2_biased_model_hard.png"
os.makedirs(os.path.dirname(savepath), exist_ok=True)
fig.savefig(savepath)
plt.show()
plt.close(fig)
print(comparator.metrics)

####################################################
#    Mitigating the impact of data distribution    #
####################################################

def M_biased_bis(x):
    return x * 0.5 + 30 + np.random.normal(0, 2, len(x))

x_gt_balanced = np.random.uniform(5, 95, 1000)
x_gt_additional_imbalance = np.clip(np.random.normal(65, 4, 1000), 5, 95)
x_gt_imbalanced = np.concatenate([x_gt_balanced, x_gt_additional_imbalance])
y_balanced = M_biased_bis(x_gt_balanced)
y_additional_imbalance = M_biased_bis(x_gt_additional_imbalance)
y_imbalanced = np.concatenate([y_balanced, y_additional_imbalance])

savepaths = []
for weighting in [None, "inverse"]:
    fig, axs = plt.subplots(figsize=(10, 8), nrows=2, ncols=2,
                            dpi=dpi_landscape,
                            height_ratios=[5, 2.5],
                            sharex=True,
                            sharey=False)
    for i, (x, y, title) in enumerate(zip([x_gt_balanced, x_gt_imbalanced],
                                          [y_balanced, y_imbalanced],
                                          ["Data distribution A", "Data distribution B"])):
        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          reference_method_type="hard",
                                          weighting=weighting)
        comparator.calculate_regression_metrics()
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   xlabel="x",
                                   ylabel="y",
                                   title="",
                                   alpha=alpha,
                                   plot_linreg=False,
                                   show_legend=False,
                                   ax=axs[0, i])
        axs[0, i].text(5, 95,
                       "MAE: {:.01f}\n".format(comparator.metrics["mae"]) +
                       r"$R^2$: {:0.2f}".format(comparator.metrics["coef_of_det_r2"]),
                       ha="left", va="top", size=13, color="red",
                       bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
        sns.histplot(x=x, ax=axs[1, i], binrange=[0, 100], bins=100)
        axs[1, i].set_ylim([0, 200])
        axs[1, i].set_ylabel("")
        axs[1, i].set_xlabel(title)

    suptitle = "No weighting" if weighting is None else "Inverse weighting"
    suptitle = "$\\bf{" + suptitle.replace(" ", "\ ") + "}$"
    if weighting is None:
        suptitle = suptitle + "\nEvaluation results dependent on the data distribution"
    else:
        suptitle = suptitle + "\nEvaluation results independent of the data distribution"
    fig.suptitle(suptitle)
    fig.tight_layout()
    savepath = "./figures/2_effect_of_inverse_weighting_{}.png".format(
        "no_weighting" if weighting is None else "inverse_weighting")
    savepaths.append(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    fig.show()
    plt.close(fig)

gif_savepath = "./figures/2_effect_of_inverse_weighting.gif"
os.makedirs(os.path.dirname(gif_savepath), exist_ok=True)

images = [Image.open(f) for f in savepaths]
images[0].save(
    gif_savepath,
    save_all=True,
    append_images=images[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0  # Infinite loop
)

####################################################
#      Focussing on the population of interest     #
####################################################

savepaths = []
for range_of_interest, title in zip([None, [14, 30]],
                                    ["No range of interest", "Using range of interest"]):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi_squared)
    comparator = ContinuousComparator(reference_method_measurements=x_gt_imbalanced,
                                      new_method_measurements=y_imbalanced,
                                      reference_method_type="hard",
                                      weighting="inverse",
                                      range_of_interest=range_of_interest)
    comparator.calculate_regression_metrics()
    comparator.plot_regression(xlim=[0, 100],
                               ylim=[0, 100],
                               xlabel="x",
                               ylabel="y",
                               title="",
                               alpha=alpha,
                               plot_linreg=False,
                               show_legend=False,
                               ax=ax)
    ax.text(5, 95,
            "MAE: {:.01f}\n".format(comparator.metrics["mae"]) +
            r"$R^2$: {:0.2f}".format(comparator.metrics["coef_of_det_r2"]),
            ha="left", va="top", size=13, color="red",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
    suptitle = "$\\bf{" + title.replace(" ", "\ ") + "}$"
    fig.suptitle(suptitle)
    fig.tight_layout()
    savepath = "./figures/2_effect_of_range_of_interest_{}.png".format(
        "no_range" if range_of_interest is None else "range")
    savepaths.append(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    fig.show()
    plt.close(fig)

gif_savepath = "./figures/2_effect_of_range_of_interest.gif"
os.makedirs(os.path.dirname(gif_savepath), exist_ok=True)

images = [Image.open(f) for f in savepaths]
images[0].save(
    gif_savepath,
    save_all=True,
    append_images=images[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0  # Infinite loop
)


####################################################
#            Assessing homoscedasticity            #
####################################################

def M_homoscedastic(x_gt):
    return x_gt + np.random.normal(0, 2, len(x_gt))
y_homoscedastic = M_homoscedastic(x_gt)

def M_heteroscedastic_monotonous(x_gt):
    y = [item + np.random.normal(0, max(0, item / 5)) for item in x_gt]
    return np.array(y)
y_heteroscedastic_monotonous = M_heteroscedastic_monotonous(x_gt)

def M_heteroscedastic_non_monotonous(x_gt):
    y = [item + np.random.normal(0, max(0, (65 - np.abs(item - 65)) / 10)) for item in x_gt]
    return np.array(y)
y_heteroscedastic_non_monotonous = M_heteroscedastic_non_monotonous(x_gt)




savepaths = []
for i, (x, y, title) in enumerate(zip([x_gt, x_gt, x_gt],
                                      [y_homoscedastic, y_heteroscedastic_monotonous, y_heteroscedastic_non_monotonous],
                                      ["Homoscedasticity",
                                       "Heteroscedasticity (monotonous)",
                                       "Heteroscedasticity (not monotonous)"])):
    fig, axs = plt.subplots(figsize=(10, 5), ncols=2,
                            dpi=dpi_landscape,
                            sharex=True,
                            sharey=False)
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="hard",
                                      weighting=weighting)
    comparator.calculate_regression_metrics()
    comparator.plot_regression(xlim=[0, 100],
                               ylim=[0, 100],
                               xlabel="x",
                               ylabel="y",
                               title="",
                               alpha=alpha,
                               plot_linreg=False,
                               show_legend=False,
                               ax=axs[0])
    comparator.plot_heteroscedasticity(xlim=[0, 100],
                                       loc="center left", # legend location
                                       ax=axs[1])
    axs
    print(title)
    print(comparator.heteroscedasticity_info)
    suptitle = "$\\bf{" + title.replace(" ", "\ ") + "}$"
    fig.suptitle(suptitle)
    fig.tight_layout()
    savepath = "./figures/2_heteroscedasticity_detector_{}.png".format(i)
    savepaths.append(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    fig.show()
    plt.close(fig)

gif_savepath = "./figures/2_heteroscedasticity_detector.gif"
os.makedirs(os.path.dirname(gif_savepath), exist_ok=True)

images = [Image.open(f) for f in savepaths]
images[0].save(
    gif_savepath,
    save_all=True,
    append_images=images[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0  # Infinite loop
)

####################################################
#   Handling uncertain reference values with BA    #
####################################################

# Simulate data with heteroscedasticity, imbalance and uncertainty in the reference values:
x = x_gt_imbalanced + np.random.normal(0, 3, len(x_gt_imbalanced))
y = M_heteroscedastic_non_monotonous(x_gt_imbalanced)

savepaths = []
for i, (weighting, range_of_interest, title) in enumerate(zip([None, None, "inverse", "inverse"],
                                                              [None, None, None, [14, 30]],
                                                              [None,
                                                               "Baseline",
                                                               "Inverse weighting",
                                                               "Inverse weighting + Range of interest"])):
    fig, ax = plt.subplots(figsize=(6, 6),
                           dpi=dpi_squared,
                           gridspec_kw={"top": 0.93})
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="soft",
                                      weighting=weighting,
                                      range_of_interest=range_of_interest)
    comparator.plot_bland_altman(xlim=[0, 100],
                                 ylim=[-50, 50],
                                 xlabel="(x + y) / 2",
                                 ylabel="y - x",
                                 title="",
                                 alpha=alpha,
                                 plot_linreg=False,
                                 show_legend=False,
                                 ax=ax)
    if title is not None:
        suptitle = "$\\bf{" + title.replace(" ", "\ ") + "}$"
        fig.suptitle(suptitle)
    fig.tight_layout()
    savepath = "./figures/2_bland_altman_{}.png".format(i)
    if i != 0:
        savepaths.append(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    fig.show()
    plt.close(fig)

gif_savepath = "./figures/2_bland_altman.gif"
os.makedirs(os.path.dirname(gif_savepath), exist_ok=True)

images = [Image.open(f) for f in savepaths]
images[0].save(
    gif_savepath,
    save_all=True,
    append_images=images[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0  # Infinite loop
)
