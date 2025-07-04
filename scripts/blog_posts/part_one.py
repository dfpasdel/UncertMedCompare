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
#   The limitations of common regression metrics   #
####################################################

# Simulate a biased model
def M_biased(x):
    return x * 0.5 + 30 + np.random.normal(0, 2, len(x))

x_gt_balanced = np.random.uniform(5, 95, 1000)
x_gt_additional_imbalance = np.clip(np.random.normal(65, 4, 1000), 5, 95)
x_gt_imbalanced = np.concatenate([x_gt_balanced, x_gt_additional_imbalance])
y_balanced = M_biased(x_gt_balanced)
y_additional_imbalance = M_biased(x_gt_additional_imbalance)
y_imbalanced = np.concatenate([y_balanced, y_additional_imbalance])

savepaths = []
for x, y, title in zip([x_gt_balanced, x_gt_imbalanced],
                       [y_balanced, y_imbalanced],
                       ["Data distribution A", "Data distribution B"]):
    fig, axs = plt.subplots(figsize=(5, 8), nrows=2,
                            dpi=dpi_portrait,
                            height_ratios=[5, 2.5],
                            sharex=True,
                            sharey=False)
    comparator = ContinuousComparator(reference_method_measurements=x,
                                      new_method_measurements=y,
                                      reference_method_type="soft")
    comparator.calculate_regression_metrics()
    comparator.plot_regression(xlim=[0, 100],
                               ylim=[0, 100],
                               xlabel="x_gt",
                               ylabel="y",
                               title="",
                               alpha=alpha,
                               plot_linreg=False,
                               show_legend=False,
                               ax=axs[0])
    axs[0].text(5, 95,
                "MAE: {:.01f}\n".format(comparator.metrics["mae"]) +
                r"$R^2$: {:0.2f}".format(comparator.metrics["coef_of_det_r2"]),
                ha="left", va="top", size=13, color="red",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
    sns.histplot(x=x, ax=axs[1], binrange=(0., 100.), bins=100)
    axs[1].set_ylim([0, 200])
    axs[1].set_ylabel("")
    axs[1].set_xlabel(title)
    fig.suptitle("MAE and R2 depend on the data distribution")
    fig.tight_layout()
    savepath = "./figures/1_mae_r2_issue_with_data_distribution_{}.png".format(title[-1])
    savepaths.append(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    fig.show()
    plt.close(fig)

gif_savepath = "./figures/1_mae_r2_issue_with_data_distribution.gif"
os.makedirs(os.path.dirname(gif_savepath), exist_ok=True)

images = [Image.open(f) for f in savepaths]
images[0].save(
    gif_savepath,
    save_all=True,
    append_images=images[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0  # Infinite loop
)

# Unbiased model with large random error
def M1(x):
    return x + np.random.normal(0, 10.18, len(x))

# Biased model with small random error
def M2(x):
    return x + 8 + np.random.normal(0, 1, len(x))

y1 = M1(x_gt)
y2 = M2(x_gt)

fig, axs = plt.subplots(figsize=(10, 5), ncols=2, dpi=dpi_landscape)
comparator1 = ContinuousComparator(reference_method_measurements=x_gt,
                                   new_method_measurements=y1)
comparator1.plot_regression(xlim=[0, 100],
                            ylim=[0, 100],
                            xlabel="x_gt",
                            ylabel="y = M1(I)",
                            title="M1",
                            alpha=alpha,
                            show_legend=False,
                            ax=axs[0])
intercept_sign = "+" if np.sign(comparator1.metrics["linreg_intercept"]) > 0 else ""
axs[0].text(5, 95,
            "MAE: {:.01f}\n".format(comparator1.metrics["mae"]) +
            r"$R^2$: {:0.2f}".format(comparator1.metrics["coef_of_det_r2"]) + "\n\n" +
            r"$\hat{y}$ = " + "{:0.2f}x".format(
                comparator1.metrics["linreg_slope"]) + intercept_sign + "{:0.2f}".format(
                comparator1.metrics["linreg_intercept"]) +
            "\n" + r"$\rho$: " + "{:0.2f}".format(comparator1.metrics["linreg_pearson_r"]),
            ha="left", va="top", size=13, color="red",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
comparator2 = ContinuousComparator(reference_method_measurements=x_gt,
                                   new_method_measurements=y2)
comparator2.plot_regression(xlim=[0, 100],
                            ylim=[0, 100],
                            xlabel="x_gt",
                            ylabel="y = M2(I)",
                            title="M2",
                            alpha=alpha,
                            show_legend=False,
                            ax=axs[1])
intercept_sign = "+" if np.sign(comparator2.metrics["linreg_intercept"]) > 0 else ""
axs[1].text(5, 95,
            "MAE: {:.01f}\n".format(comparator2.metrics["mae"]) +
            r"$R^2$: {:0.2f}".format(comparator2.metrics["coef_of_det_r2"]) + "\n\n" +
            r"$\hat{y}$ = " + "{:0.2f}x".format(
                comparator2.metrics["linreg_slope"]) + intercept_sign + "{:0.2f}".format(
                comparator2.metrics["linreg_intercept"]) +
            "\n" + r"$\rho$: " + "{:0.2f}".format(comparator2.metrics["linreg_pearson_r"]),
            ha="left", va="top", size=13, color="red",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
fig.tight_layout()
savepath = "./figures/1_mae_issue.png"
os.makedirs(os.path.dirname(savepath), exist_ok=True)
fig.savefig(savepath)
plt.show()
plt.close(fig)

####################################################
#       Regression analysis considerations         #
####################################################

def calculate_coef_of_det_and_pearson_correlation(x, y):
    coef_of_det = metrics.r2_score(x, y)
    pearson_correlation = pearsonr(x, y)[0]
    print("Coef. of determination: {:.05f}\nPearson's correlation squared: {:.05f}\n\n".format(
        coef_of_det,
        pearson_correlation ** 2))
# Calculate the regression line between X and Y:
slope, intercept, _, _, _ = linregress(x_gt, y2)
y2_hat = x_gt * slope + intercept

# Y vs. X
calculate_coef_of_det_and_pearson_correlation(x_gt, y2)
# Y_hat vs. Y
calculate_coef_of_det_and_pearson_correlation(y2, y2_hat)


####################################################
#     The impact of uncertain reference values     #
####################################################

# Add some uncertainty to the ground truth age:
x = x_gt + np.random.normal(0, 5, len(x_gt))
# Simulate a model M that returns the ground truth:
y = x_gt

fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi_squared)
comparator = ContinuousComparator(reference_method_measurements=x,
                                  new_method_measurements=y,
                                  reference_method_type="hard")
comparator.plot_regression(xlim=[0, 100],
                           ylim=[0, 100],
                           xlabel="x",
                           ylabel="y = M(I)",
                           title="",
                           alpha=alpha,
                           show_legend=False,
                           ax=ax)
intercept_sign = "+" if np.sign(comparator.metrics["linreg_intercept"]) > 0 else ""
ax.text(5, 95,
        "MAE: {:.01f}\n".format(comparator.metrics["mae"]) +
        r"$R^2$: {:0.2f}".format(comparator.metrics["coef_of_det_r2"]) + "\n\n" +
        r"$\hat{y}$ = " + "{:0.2f}x".format(
            comparator.metrics["linreg_slope"]) + intercept_sign + "{:0.2f}".format(
            comparator.metrics["linreg_intercept"]) +
        "\n" + r"$\rho$: " + "{:0.2f}".format(comparator.metrics["linreg_pearson_r"]),
        ha="left", va="top", size=13, color="black",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
fig.tight_layout()
savepath = "./figures/1_slope_with_uncertain_ref.png"
os.makedirs(os.path.dirname(savepath), exist_ok=True)
fig.savefig(savepath)
plt.show()
plt.close(fig)

####################################################
#   Bland-Altman analysis: A better alternative    #
####################################################

# Add some uncertainty to the ground truth age:
x = x_gt + np.random.normal(0, 7, len(x_gt))
# Simulate a biased model:
y = M_biased(x_gt)

# Bland-Altman plotting
comparator = ContinuousComparator(reference_method_measurements=x,
                                  new_method_measurements=y,
                                  reference_method_type="soft")
fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi_squared)
comparator.plot_bland_altman(xlim=[0, 100],
                             ylim=[-50, 50],
                             xlabel="(x + y) / 2",
                             ylabel="y - x",
                             title="",
                             alpha=alpha,
                             ax=ax)
fig.tight_layout()
savepath = "./figures/1_bland_altman_example.png"
os.makedirs(os.path.dirname(savepath), exist_ok=True)
fig.savefig(savepath)
plt.show()
plt.close(fig)

# Effect of heteroscedasticity coupled with imbalance on Bland-Altman

# Simulate an heteroscedastic model
def M_heteroscedastic(x):
    y = [item + np.random.normal(0, max(0, item / 8)) for item in x]
    return np.array(y)
    
# Simulate balanced and imbalanced ground truths
x_gt_balanced = np.random.uniform(0, 100, 1000)
x_gt_additional_imbalance = np.clip(np.random.normal(75, 7, 3000), 0, 100)
x_gt_imbalanced = np.concatenate([x_gt_balanced, x_gt_additional_imbalance])

# Add uncertainty to the ground truth values
x_balanced = x_gt_balanced + np.random.normal(0, 2, len(x_gt_balanced))
x_additional_imbalance = x_gt_additional_imbalance + np.random.normal(0, 2, len(x_gt_additional_imbalance))
x_imbalanced = np.concatenate([x_balanced, x_additional_imbalance])

# Generate estimates with an heteroscedastic model
y_balanced_heteroscedastic = M_heteroscedastic(x_gt_balanced)
y_additional_imbalance_heteroscedastic = M_heteroscedastic(x_gt_additional_imbalance)
y_imbalanced_heteroscedastic = np.concatenate([y_balanced_heteroscedastic, y_additional_imbalance_heteroscedastic])

savepaths = []
for x, y, title in zip([x_balanced, x_imbalanced],
                       [y_balanced_heteroscedastic, y_imbalanced_heteroscedastic],
                       ["Data distribution A", "Data distribution B"]):
    fig, axs = plt.subplots(figsize=(5, 8), nrows=2,
                            dpi=dpi_portrait,
                            height_ratios=[5, 2.5],
                            sharex=True,
                            sharey=False)
    comparator_hetero = ContinuousComparator(reference_method_measurements=x,
                                             new_method_measurements=y,
                                             reference_method_type="soft")
    comparator_hetero.calculate_regression_metrics()
    comparator_hetero.plot_bland_altman(xlim=[0, 100],
                                        ylim=[-50, 50],
                                        xlabel="(x + y) / 2",
                                        ylabel="y - x",
                                        title="",
                                        alpha=alpha,
                                        plot_linreg=False,
                                        show_legend=False,
                                        ax=axs[0])
    axs[0].text(5, 45,
                "MAE: {:.01f}\n".format(comparator_hetero.metrics["mae"]) +
                r"$R^2$: {:0.2f}".format(comparator_hetero.metrics["coef_of_det_r2"]),
                ha="left", va="top", size=13, color="red",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=5))
    means = comparator_hetero.mean_values
    sns.histplot(x=means, ax=axs[1], binrange=[0, 100], bins=100)
    axs[1].set_ylim([0, 200])
    axs[1].set_ylabel("")
    axs[1].set_xlabel(title)
    fig.suptitle("Different data distributions in the\npresence of heteroscedasticity")
    fig.tight_layout()
    savepath = "./figures/1_heteroscedasticity_issue_{}.png".format(title[-1])
    savepaths.append(savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    fig.show()
    plt.close(fig)

gif_savepath = "./figures/1_heteroscedasticity_issue.gif"
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
#      Avoiding misuse of AUC in regression        #
####################################################

def write_tp_tn_fp_fn(ax, x, y, threshold, fontsize=12):
    tp = np.sum(np.where((x < threshold) & (y < threshold), 1, 0))
    tn = np.sum(np.where((x >= threshold) & (y >= threshold), 1, 0))
    fp = np.sum(np.where((x >= threshold) & (y < threshold), 1, 0))
    fn = np.sum(np.where((x < threshold) & (y >= threshold), 1, 0))
    s = tp + tn + fp + fn
    t = ax.text(3, 3, "TP: {:.01f}%".format(100 * tp / s), fontsize=fontsize,
                ha="left", va="bottom")
    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0, pad=2))
    t = ax.text(97, 97, "TN: {:.01f}%".format(100 * tn / s), fontsize=fontsize,
                ha="right", va="top")
    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0, pad=2))
    t = ax.text(97, 3, "FP: {:.01f}%".format(100 * fp / s), fontsize=fontsize,
                ha="right", va="bottom")
    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0, pad=2))
    t = ax.text(3, 97, "FN: {:.01f}%".format(100 * fn / s), fontsize=fontsize,
                ha="left", va="top")
    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0, pad=2))

def calculate_auc_from_regression(labels, preds, cutoff, axs):
    # Display the estimations
    axs[0].scatter(labels, preds, alpha=alpha)
    axs[0].set_xlim([0, 100])
    axs[0].set_ylim([0, 100])
    axs[0].set_xlabel("x_gt")
    axs[0].set_ylabel("y = M(I)")
    axs[0].set_aspect("equal")
    write_tp_tn_fp_fn(ax=axs[0],
                      x=labels,
                      y=preds,
                      threshold=cutoff)
    axs[0].axhline(y=cutoff, xmin=0, xmax=1, ls="--", color="black",
                   label="AUC classification\nthreshold ({:02d}%)".format(cutoff))
    axs[0].axvline(x=cutoff, ymin=0, ymax=1, ls="--", color="black")
    # Display AUC:
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels > cutoff, y_score=preds)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot(ax=axs[1])
    axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[1].set_aspect("equal")
    axs[1].legend(facecolor="white", frameon=True, loc="lower right")
    fig.tight_layout()


y = x_gt + np.random.normal(0, 4, len(x_gt))
savepaths = []
for restrict_range in [False, True]:
    fig, axs = plt.subplots(figsize=(10, 5), ncols=2, dpi=dpi_landscape)
    calculate_auc_from_regression(labels=x_gt[(x_gt > 14) & (x_gt < 30)] if restrict_range else x_gt ,
                                  preds=y[(x_gt > 14) & (x_gt < 30)] if restrict_range else y,
                                  cutoff=22,
                                  axs=axs)
    fig.tight_layout()
    savepath = f"./figures/1_AUC_issue_{restrict_range}.png"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath)
    savepaths.append(savepath)
    plt.show()
    plt.close(fig)

gif_savepath = "./figures/1_AUC_issue.gif"
os.makedirs(os.path.dirname(gif_savepath), exist_ok=True)

images = [Image.open(f) for f in savepaths]

images[0].save(
    gif_savepath,
    save_all=True,
    append_images=images[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0  # Infinite loop
)
