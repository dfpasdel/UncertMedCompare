import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.weightstats import DescrStatsW
from typing import Optional

from UncertMedCompare.utilities.misc import weighted_error_mean_and_std, rolling_mean
from UncertMedCompare.utilities.plot_utilities import boldify_legend_text, get_regression_line_soft_range_of_interest_intersections
from UncertMedCompare.utilities.math_utilities import nearest_ceiling_idx
from UncertMedCompare.config import DEFAULT_STYLE


class ContinuousComparator(object):
    """
    Class for comparison of two measurement methods that take value on a continuous scale

    Attributes
    ----------
        metrics
            Dictionary containing the metrics calculated so far when calling the different methods
        heteroscedasticity_info
            Dictionary containing the results of the heteroscedasticity tests

    Methods
    -------
        fit_linear_regression()
            Calculate all linear regression parameters

        fit_bland_altman()
            Calculates mean error and limits of agreement

        calculate_error_metrics()
            Calculates mean error and standard deviation

        calculate_regression_metrics()
            Calculates MAE, R2 (coefficient of determination), MSE, RMSE

        bootstrap_error_metrics()
            Calculates mean error and standard deviation with bootstrapped CIs

        bootstrap_regression_metrics()
            Calculates MAE, R2 (coefficient of determination), MSE, RMSE with bootstrapped CIs
    """
    def __init__(self,
                 reference_method_measurements: np.ndarray,
                 new_method_measurements: np.ndarray,
                 reference_method_type: str = "hard",
                 range_of_interest: Optional[list] = None,
                 weighting: Optional[str] = None,
                 binwidth: Optional[float] = None,
                 min_samples_by_bin: int = 1,
                 limit_of_agreement: float = 1.96,
                 confidence_interval: Optional[int] = None,
                 bootstrap_samples: int = 10000,
                 units: Optional[str] = None,
                 format_dimensional: str = "{:.02f}",
                 format_non_dimensional: str = "{:.02f}",
                 significance_level: float = 0.05,
                 soft_regression_method: str = "LP"):
        """
        Initializes the ContinuousComparator object with the required parameters for configuring the analysis, including
        measurements by the reference method, measurements by the new method, weighting, binning, statistical
        thresholds, and formatting.

        Parameters
        ----------
        reference_method_measurements : np.ndarray
            Measurement values from the reference method.

        new_method_measurements : np.ndarray
            Measurement values from the new method to be evaluated.

        reference_method_type : str, optional, default: "hard"
            Whether the measurement values from the reference method are attended by some error. Possible options:
            - "hard": There is no uncertainty in the reference measurements (they are perfectly reproducible, no error).
            - "soft": Reference measurements are uncertain (measuring two times will give two different values, error).

        range_of_interest : list, optional, default: None
            Specifies the range of values to focus the analysis on.
            Only data within this range is considered.

        weighting : str, optional, default: None
            Specifies the weighting method for observations. Possible options include:
            - None: No weighting is applied.
            - "inverse": Weight based on frequency.

        binwidth : float, optional, default: None
            The width of bins for dividing data into discrete ranges for inverse weighting.

        min_samples_by_bin : int, optional, default: 1
            The minimum number of samples required per bin to ensure robust weighting.

        limit_of_agreement : float, optional, default: 1.96
            The multiplier for computing the limits of agreement (e.g., in Bland-Altman plots).
            Default is 1.96, which corresponds to approximately 95% coverage for normally distributed data.

        confidence_interval : int, optional, default: None
            The confidence interval percentage for statistical estimates.
            Example: 95 for a 95% confidence interval.

        bootstrap_samples : int, optional, default: 10000
            Number of bootstrap samples used in analyses.

        units : str, optional, default: None
            The unit of the measurement values (e.g. "mL").

        format_dimensional : str, optional, default: "{:.02f}"
            Format string for displaying dimensional values (e.g., "12.34 mL").

        format_non_dimensional : str, optional, default: "{:.02f}"
            Format string for displaying non-dimensional values (e.g., "0.87").

        significance_level : float, optional, default: 0.05
            The significance level for hypothesis testing.

        soft_regression_method : str, optional, default: "LP"
            Specifies the regression method used when the reference values are uncertain (soft). Possible options
            include:
            - "LP": Least Products (https://pubmed.ncbi.nlm.nih.gov/20337658/)
            - "BA": Bland-Altman
        """
        self.ref_values = np.asarray(reference_method_measurements)
        self.new_values = np.asarray(new_method_measurements)
        self.reference_method_type = reference_method_type
        if self.reference_method_type not in ['soft', 'hard']:
            raise ValueError("The reference_method_type argument for ContinuousComparator must either be hard (when the "
                             "reference values are certain) or soft (when the reference values are uncertain)")
        self.soft_regression_method = soft_regression_method
        if self.reference_method_type == "hard":
            self.regression_method_name = "Least squares"
        elif self.reference_method_type == "soft":
            if self.soft_regression_method == "LP":
                self.regression_method_name = "Least products"
            elif self.soft_regression_method == "BA":
                self.regression_method_name = "Bland-Altman"
            else:
                raise ValueError("The soft_regression_method argument for ContinuousComparator must either be LP (least"
                                 " products) or BA (Bland-Altman)")
        self.range_of_interest = range_of_interest
        self.weighting = weighting
        if self.weighting is None:
            self._weighting_method = "no weighting"
        elif self.weighting == "inverse":
            self._weighting_method = "inverse weighting"
        self.binwidth = binwidth
        if not self.binwidth:
            self.binwidth = (np.max([self.ref_values, self.new_values]) - np.min(
                [self.ref_values, self.new_values])) / 100
        self.min_samples_by_bin = min_samples_by_bin
        self.limit_of_agreement = limit_of_agreement
        self.confidence_interval = confidence_interval
        self.bootstrap_samples = bootstrap_samples
        self.prev_rc_params = plt.rcParams.copy()
        self.units = units
        self.format_dimensional = format_dimensional
        self.format_non_dimensional = format_non_dimensional
        self.significance_level = significance_level

        #
        # The below lines should be equivalent to self.reset():
        #

        # Property attributes:
        self._mean_values = None
        self._diff_values = None
        self._w_ref_values = None
        self._w_new_values = None
        self._w_mean_values = None
        self._ref_values_of_interest = None
        self._new_values_of_interest = None
        self._mean_values_of_interest = None
        self._diff_values_of_interest = None
        self._calibration_function = None
        self._heteroscedasticity_info = None

        # Not property attributes
        self._ref_values_hist = None
        self._new_values_hist = None
        self._mean_values_hist = None
        self._ref_values_hist_smoothed = None
        self._new_values_hist_smoothed = None
        self._mean_values_hist_smoothed = None
        self._ref_values_hist_bin_edges = None
        self._new_values_hist_bin_edges = None
        self._mean_values_hist_bin_edges = None
        self._ref_values_rolling_mean_window = None
        self._new_values_rolling_mean_window = None
        self._mean_values_rolling_mean_window = None
        self._range_of_interest_ref_values = None
        self._range_of_interest_new_values = None
        self._range_of_interest_mean_values = None
        self._linreg_possible = True
        self._ba_possible = True
        self.metrics = {}
        self._has_error_metrics = False
        self._has_error_metrics_bootstrap_ci = False
        self._has_regression_metrics = False
        self._has_regression_metrics_bootstrap_ci = False
        self._has_linear_regression_calculated = False
        self._has_bland_altman_calculated = False
        self._h_residuals = None
        self._h_residuals_of_interest = None
        self._ref_values_bounds = None
        self._new_values_bounds = None
        self._mean_values_bounds = None

        self.set_data_range()

    @property
    def mean_values(self):
        if self._mean_values is None:
            self._mean_values = 0.5 * (self.ref_values + self.new_values)
        return self._mean_values

    @property
    def diff_values(self):
        if self._diff_values is None:
            self._diff_values = self.new_values - self.ref_values
        return self._diff_values

    @staticmethod
    def get_histogram(a, binwidth, base_range=None):
        """
        Method that generates histogram based on binwidth argument instead of bins and range (like numpy does)
        """
        if base_range is not None:
            low_bound = min(min(a), min(base_range))
            high_bound = max(max(a), max(base_range))
        else:
            low_bound = min(a)
            high_bound = max(a)
        range = [(low_bound // binwidth) * binwidth,
                 (high_bound // binwidth + 1) * binwidth if high_bound // binwidth != high_bound / binwidth
                 else (high_bound // binwidth) * binwidth
                 ]
        bins = int((max(range) - min(range)) / binwidth)
        return np.histogram(a, bins=bins, range=range)

    @property
    def w_ref_values(self):
        if self._w_ref_values is None:
            if self.weighting is None or len(self.ref_values_of_interest) == 0:
                self._w_ref_values = None
            elif self.weighting == "inverse":
                if self.range_of_interest and self.reference_method_type == "soft":
                    # When using range_of_interest and soft reference values, the histogram within the roi is different
                    # for ref_values and for ref_values_of_interest since the reference values of interest are chosen
                    # based on the mean_value being within the range of interest.
                    # The rolling mean may create some boundaries issues.
                    hist, edges = self.get_histogram(self.ref_values_of_interest,
                                                     binwidth=self.binwidth,
                                                     base_range=self.range_of_interest)
                else:
                    hist, edges = self.get_histogram(self.ref_values,
                                                     binwidth=self.binwidth,
                                                     base_range=self.range_of_interest)
                self._ref_values_hist = hist
                self._ref_values_hist_bin_edges = edges[:-1]
                N = 1
                while N < len(edges):
                    smoothed = rolling_mean(hist, N)
                    if self.range_of_interest:
                        idx_start = np.argmax(np.where(edges < min(self._range_of_interest_ref_values), 0, 1)) - 1
                        # Avoid bins_to_check[0] to be -1 when the lower bound of the range of interest is smaller than
                        # the lowest value
                        idx_start = max(0, idx_start)
                        idx_end = np.argmax(np.where(edges >= max(self._range_of_interest_ref_values), 1, 0))
                        if idx_end == 0:
                            # Avoid bins_to_check[1] to be 0 when max(bin_edges) < max(range_to_check)
                            idx_end = len(hist)
                        smoothed_to_test = smoothed[idx_start:idx_end]
                    else:
                        smoothed_to_test = smoothed
                    if np.min(smoothed_to_test) >= self.min_samples_by_bin:
                        self._ref_values_rolling_mean_window = N * self.binwidth
                        self._ref_values_hist_smoothed = smoothed
                        self._w_ref_values = np.array([1. / smoothed[nearest_ceiling_idx(edges, value)] for value in
                                                   self.ref_values_of_interest])
                        break
                    N += 2
                if self._w_ref_values is None:
                    warnings.warn("Could not calculate the ref_values inverse weights.")
            else:
                raise ValueError("Weighting strategy {} is not implemented".format(self.weighting))
        return self._w_ref_values

    @property
    def w_new_values(self):
        if self._w_new_values is None:
            if self.weighting is None or len(self.new_values_of_interest) == 0:
                self._w_new_values = None
            elif self.weighting == "inverse":
                if self.range_of_interest and self.reference_method_type == "soft":
                    # When using range_of_interest and soft reference values, the histogram within the roi is different
                    # for new_values and for new_values_of_interest since the new values of interest are chosen based on
                    # the mean_value being within the range of interest.
                    # The rolling mean may create some boundaries issues.
                    hist, edges = self.get_histogram(self.new_values_of_interest,
                                                     binwidth=self.binwidth,
                                                     base_range=self.range_of_interest)
                else:
                    hist, edges = self.get_histogram(self.new_values,
                                                     binwidth=self.binwidth,
                                                     base_range=self.range_of_interest)
                self._new_values_hist = hist
                self._new_values_hist_bin_edges = edges[:-1]
                N = 1
                while N < len(edges):
                    smoothed = rolling_mean(hist, N)
                    if self.range_of_interest:
                        idx_start = np.argmax(np.where(edges < min(self._range_of_interest_new_values), 0, 1)) - 1
                        # Avoid bins_to_check[0] to be -1 when the lower bound of the range of interest is smaller than
                        # the lowest value
                        idx_start = max(0, idx_start)
                        idx_end = np.argmax(np.where(edges >= max(self._range_of_interest_new_values), 1, 0))
                        if idx_end == 0:
                            # Avoid bins_to_check[1] to be 0 when max(bin_edges) < max(range_to_check)
                            idx_end = len(hist)
                        smoothed_to_test = smoothed[idx_start:idx_end]
                    else:
                        smoothed_to_test = smoothed
                    if np.min(smoothed_to_test) >= self.min_samples_by_bin:
                        self._new_values_rolling_mean_window = N * self.binwidth
                        self._new_values_hist_smoothed = smoothed
                        self._w_new_values = np.array(
                            [1. / smoothed[nearest_ceiling_idx(edges, value)] for value in self.new_values_of_interest])
                        break
                    N += 2
                if self._w_new_values is None:
                    warnings.warn("Could not calculate the new_values inverse weights.")
            else:
                raise ValueError("Weighting strategy {} is not implemented".format(self.weighting))
        return self._w_new_values

    @property
    def w_mean_values(self):
        if self._w_mean_values is None:
            if self.weighting is None or len(self.mean_values_of_interest) == 0:
                self._w_mean_values = None
            elif self.weighting == "inverse":
                hist, edges = self.get_histogram(self.mean_values,
                                                 binwidth=self.binwidth,
                                                 base_range=self.range_of_interest)
                self._mean_values_hist = hist
                self._mean_values_hist_bin_edges = edges[:-1]
                N = 1
                while N < len(edges):
                    smoothed = rolling_mean(hist, N)
                    if self.range_of_interest:
                        idx_start = np.argmax(np.where(edges < min(self._range_of_interest_mean_values), 0, 1)) - 1
                        # Avoid bins_to_check[0] to be -1 when the lower bound of the range of interest is smaller than
                        # the lowest value
                        idx_start = max(0, idx_start)
                        idx_end = np.argmax(np.where(edges >= max(self._range_of_interest_mean_values), 1, 0))
                        if idx_end == 0:
                            # Avoid bins_to_check[1] to be 0 when max(bin_edges) < max(range_to_check)
                            idx_end = len(hist)
                        smoothed_to_test = smoothed[idx_start:idx_end]
                    else:
                        smoothed_to_test = smoothed
                    if np.min(smoothed_to_test) >= self.min_samples_by_bin:
                        self._mean_values_rolling_mean_window = N * self.binwidth
                        self._mean_values_hist_smoothed = smoothed
                        self._w_mean_values = np.array(
                            [1. / smoothed[nearest_ceiling_idx(edges, value)] for value in self.mean_values_of_interest])
                        break
                    N += 2
                if self._w_mean_values is None:
                    warnings.warn("Could not calculate the mean_values inverse weights.")
            else:
                raise ValueError("Weighting strategy {} is not implemented".format(self.weighting))
        return self._w_mean_values

    def bootstrap_metric(self, ref_values, new_values, func, sample_weight=None, **kwargs):
        if self.confidence_interval is not None:
            if not (self.confidence_interval < 99.9) & (self.confidence_interval > 1):
                raise ValueError(
                    f'"confidence_interval" must be a number in the range 1 to 99, '
                    f'"{self.confidence_interval}" provided.')
        else:
            raise ValueError("confidence_interval must be specified to bootstrap metrics")
        if self.bootstrap_samples < 100:
            raise ValueError("Please specify a number of bootstraps larger than 100.")
        left = (1 - self.confidence_interval / 100) / 2
        right = 1 - left

        ref_values = np.asarray(ref_values)
        new_values = np.asarray(new_values)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        bootstraps = []
        w = None
        for _ in range(self.bootstrap_samples):
            ind = np.random.choice(len(ref_values), len(ref_values))
            if sample_weight is not None:
                # This is an approximation, the weights should ideally be recalculated, but this is too computational
                # intensive. Assumes that the distribution does not change significantly when drawing new samples.
                w = sample_weight[ind]
            bootstraps.append(func(ref_values[ind], new_values[ind], sample_weight=w, **kwargs))
        bootstraps = np.array(bootstraps)
        if np.isnan(np.sum(bootstraps)):
            bootstraps *= np.nan
        bootstraps = np.sort(bootstraps, axis=0)
        if len(bootstraps.shape) == 1:
            return func(ref_values, new_values, sample_weight=sample_weight, **kwargs), \
                   bootstraps[round(left * self.bootstrap_samples)], \
                   bootstraps[round(right * self.bootstrap_samples)]
        else:
            result = []
            values = func(ref_values, new_values, sample_weight=sample_weight, **kwargs)
            for idx in range(bootstraps.shape[1]):
                result.append([values[idx],
                               bootstraps[round(left * self.bootstrap_samples), idx],
                               bootstraps[round(right * self.bootstrap_samples), idx]])
            return result

    def set_data_range(self):
        if self.range_of_interest:
            if self.reference_method_type == "soft":
                indexes_linreg = (self.mean_values <= max(self.range_of_interest)) \
                                 & (self.mean_values >= min(self.range_of_interest))
            elif self.reference_method_type == "hard":
                indexes_linreg = (self.ref_values <= max(self.range_of_interest)) \
                                 & (self.ref_values >= min(self.range_of_interest))
            indexes_ba = (self.mean_values <= max(self.range_of_interest)) \
                         & (self.mean_values >= min(self.range_of_interest))
            if True not in indexes_linreg:
                self._linreg_possible = False
            if True not in indexes_ba:
                self._ba_possible = False
            self._ref_values_of_interest = self.ref_values[indexes_linreg]
            self._new_values_of_interest = self.new_values[indexes_linreg]
            self._mean_values_of_interest = self.mean_values[indexes_ba]
            self._diff_values_of_interest = self.diff_values[indexes_ba]
            if self.reference_method_type == "soft" and self._linreg_possible:
                self._range_of_interest_ref_values = [min(self._ref_values_of_interest), max(self._ref_values_of_interest)]
                self._range_of_interest_new_values = [min(self._new_values_of_interest), max(self._new_values_of_interest)]
            elif self.reference_method_type == "hard":
                self._range_of_interest_ref_values = self.range_of_interest
                self._range_of_interest_new_values = self.range_of_interest
            self._range_of_interest_mean_values = self.range_of_interest
            self._ref_values_bounds = self._range_of_interest_ref_values
            self._new_values_bounds = self._range_of_interest_new_values
            self._mean_values_bounds = self._range_of_interest_mean_values
        else:
            if self.weighting == "inverse":
                warnings.warn("Inversely weighting samples without specifying the range of interest may cause issues "
                              "due to few values at the sides of the data distribution. Check the distribution plot "
                              "before reporting results or consider setting the range of interest.")
            self._ref_values_of_interest = self.ref_values
            self._new_values_of_interest = self.new_values
            self._mean_values_of_interest = self.mean_values
            self._diff_values_of_interest = self.diff_values
            self._range_of_interest_ref_values = self.range_of_interest
            self._range_of_interest_new_values = self.range_of_interest
            self._range_of_interest_mean_values = self.range_of_interest
            self._ref_values_bounds = [min(self._ref_values_of_interest), max(self._ref_values_of_interest)]
            self._new_values_bounds = [min(self._new_values_of_interest), max(self._new_values_of_interest)]
            self._mean_values_bounds = [min(self._mean_values_of_interest), max(self._mean_values_of_interest)]

    @property
    def ref_values_of_interest(self):
        if self._ref_values_of_interest is None:
            self.set_data_range()
        return self._ref_values_of_interest

    @property
    def new_values_of_interest(self):
        if self._new_values_of_interest is None:
            self.set_data_range()
        return self._new_values_of_interest

    @property
    def mean_values_of_interest(self):
        if self._mean_values_of_interest is None:
            self.set_data_range()
        return self._mean_values_of_interest

    @property
    def diff_values_of_interest(self):
        if self._diff_values_of_interest is None:
            self.set_data_range()
        return self._diff_values_of_interest

    @property
    def heteroscedasticity_info(self):
        if self._heteroscedasticity_info is None:
            self.check_heteroscedasticity()
        return self._heteroscedasticity_info

    def fit_linear_regression(self):
        if self._linreg_possible and not self._has_linear_regression_calculated:
            if self.reference_method_type == "hard":
                # Weighted Least Squares
                self.metrics["linreg_intercept"], self.metrics["linreg_slope"], self.metrics["linreg_pearson_r"], \
                self.metrics["linreg_intercept_std"], self.metrics["linreg_slope_std"] = \
                    wls_wrapper(x=self.ref_values_of_interest,
                                y=self.new_values_of_interest,
                                sample_weight=self.w_ref_values,
                                return_std=True)
            elif self.reference_method_type == "soft":
                if self.soft_regression_method == "LP":
                    # Weighted Least Products (Weighted Reduced Major Axis)
                    self.metrics["linreg_intercept"], self.metrics["linreg_slope"], self.metrics["linreg_pearson_r"] = \
                        wlp_wrapper(x=self.ref_values_of_interest, y=self.new_values_of_interest,
                                    sample_weight_x=self.w_ref_values,
                                    sample_weight_y=self.w_new_values)
                elif self.soft_regression_method == "BA":
                    # Bland-Altman approach
                    # Regress differences on means and rotate 45 degrees
                    self.fit_bland_altman()
                    ba_slope = self.metrics["ba_slope"]
                    ba_intercept = self.metrics["ba_intercept"]
                    self.metrics["linreg_slope"] = (2 + ba_slope) / (2 - ba_slope)
                    self.metrics["linreg_intercept"] = 2 * ba_intercept / (2 - ba_slope)
                    self.metrics["linreg_pearson_r"] = DescrStatsW(data=np.array([self.ref_values_of_interest,
                                                                                  self.new_values_of_interest]).T,
                                                                   weights=self.w_mean_values).corrcoef[0][1]
            self._has_linear_regression_calculated = True

    def fit_bland_altman(self):
        if self.reference_method_type == "hard":
            raise ValueError("Bland-Altman is not to be used with hard reference values (the whole point of "
                             "Bland-Altman is to account for variability in the reference values)")

        if self._ba_possible and not self._has_bland_altman_calculated:
            # Calculate the trend:
            # Use WLS ("hard reference") as the X-axis is the mean of the two methods
            if self.confidence_interval is not None:
                [self.metrics["ba_intercept"],
                 self.metrics["ba_intercept_low_bound"],
                 self.metrics["ba_intercept_up_bound"]], \
                [self.metrics["ba_slope"],
                 self.metrics["ba_slope_low_bound"],
                 self.metrics["ba_slope_up_bound"]], \
                [self.metrics["ba_corr"],
                 self.metrics["ba_corr_low_bound"],
                 self.metrics["ba_corr_up_bound"]] = self.bootstrap_metric(self.mean_values_of_interest,
                                                                           self.diff_values_of_interest,
                                                                           sample_weight=self.w_mean_values,
                                                                           func=wls_wrapper)
            else:
                self.metrics["ba_intercept"], \
                self.metrics["ba_slope"], \
                self.metrics["ba_corr"] = wls_wrapper(self.mean_values_of_interest,
                                                      self.diff_values_of_interest,
                                                      sample_weight=self.w_mean_values)
                self.metrics["ba_intercept_low_bound"], \
                self.metrics["ba_intercept_up_bound"], \
                self.metrics["ba_slope_low_bound"], \
                self.metrics["ba_slope_up_bound"], \
                self.metrics["ba_corr_low_bound"], \
                self.metrics["ba_corr_up_bound"] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Calculate mean and std error
            self.calculate_error_metrics()
            if self.confidence_interval is not None:
                # Calculate confidence intervals
                self.bootstrap_error_metrics()
                self.metrics["ba_loa"] = self.limit_of_agreement * self.metrics["std_error"]
            self._has_bland_altman_calculated = True
        elif not self._has_bland_altman_calculated:
            [self.metrics["ba_intercept"],
             self.metrics["ba_intercept_low_bound"],
             self.metrics["ba_intercept_up_bound"]], \
            [self.metrics["ba_slope"],
             self.metrics["ba_slope_low_bound"],
             self.metrics["ba_slope_up_bound"]] = [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]

            self.calculate_error_metrics()
            if self.confidence_interval is not None:
                self.bootstrap_error_metrics()
                self.metrics["ba_loa"] = self.limit_of_agreement * self.metrics["std_error"]
            self._has_bland_altman_calculated = True

    def check_heteroscedasticity(self):
        """
        Function to check heteroscedasticity with White and Breusch-Pagan tests.
        H0: homoscedasticity, H1: heteroscedasticity
        Breusch-Pagan detects monotonous heteroscedasticity, whereas White test can detect non-monotonous
        heteroscedasticity.
        For hard reference values, new measurement values are regressed on reference measurement values. For soft
        reference values, differences are regressed on mean values.
        Note: It is important to have a qualitative assessment of heteroscedasticity in addition to p-values analysis
        """
        self.fit_linear_regression()
        if self.reference_method_type == "hard":
            self._h_residuals = self.new_values - \
                                (self.metrics["linreg_slope"] * self.ref_values + self.metrics["linreg_intercept"])
            self._h_residuals_of_interest = self.new_values_of_interest - \
                                            (self.metrics["linreg_slope"] * self.ref_values_of_interest +
                                             self.metrics["linreg_intercept"])
        elif self.reference_method_type == "soft":
            intercept, slope, _ = wls_wrapper(self.mean_values_of_interest, self.diff_values_of_interest,
                                              sample_weight=self.w_mean_values)
            self._h_residuals = self.diff_values - \
                                (slope * self.mean_values + intercept)
            self._h_residuals_of_interest = self.diff_values_of_interest - \
                                            (slope * self.mean_values_of_interest + intercept)
        resid = self._h_residuals_of_interest
        exog = self.ref_values_of_interest
        # _, _, _, p_value, _ = stats.linregress(exog, np.abs(resid))
        exog = np.concatenate([[np.ones(len(exog))], [exog]], axis=0).T
        _, p_value_white, _, _ = het_white(resid=resid, exog=exog)
        _, p_value_breuschpagan, _, _ = het_breuschpagan(resid=resid, exog_het=exog)
        info = {}
        info["white"] = True if p_value_white < self.significance_level else False
        info["p_value_white"] = p_value_white
        info["breuschpagan"] = True if p_value_breuschpagan < self.significance_level else False
        info["p_value_breuschpagan"] = p_value_breuschpagan
        self._heteroscedasticity_info = info

    @property
    def calibration_function(self):
        """
        Returns a calibration function to be applied on the new measurement values.
        The intented use is to: 1) Calculate the function coefficients on the validation set and 2) apply the function
        on all external test sets and prospective inference data.
        """
        if self._calibration_function is None:
            if self.reference_method_type == "soft":
                # The calibration is done from a Bland-Altman plot perspective, and de-trends the Bland-Altman plot
                non_calibrated_new_values = self.new_values.copy()
                slope_corrections = []
                intercept_corrections = []
                if self.range_of_interest is not None:
                    # Need to do it recursively as calibration adds/removes data points within the range of interest
                    for _ in range(20):
                        intercept, slope, _ = wls_wrapper(self.mean_values_of_interest, self.diff_values_of_interest,
                                                          sample_weight=self.w_mean_values)
                        slope_correction = (1 - slope / 2) / (1 + slope / 2)
                        intercept_correction = - intercept / (1 + slope / 2)
                        self.new_values = self.new_values * slope_correction + intercept_correction
                        slope_corrections.append(slope_correction)
                        intercept_corrections.append(intercept_correction)
                        self.reset()
                        if (slope == 1.) and (intercept == 0.):
                            break
                    # NOTE: The iterative process gets sometimes stuck into a periodic pattern,
                    # This should be detected
                else:
                    intercept, slope, _ = wls_wrapper(x=self.mean_values_of_interest,
                                                      y=self.diff_values_of_interest,
                                                      sample_weight=self.w_mean_values)
                    slope_correction = (1 - slope / 2) / (1 + slope / 2)
                    intercept_correction = - intercept / (1 + slope / 2)
                    slope_corrections.append(slope_correction)
                    intercept_corrections.append(intercept_correction)
                self.new_values = non_calibrated_new_values
                self.reset()

                def calibration_function_ba(new_values):
                    for slope_correction, intercept_correction in zip(slope_corrections, intercept_corrections):
                        new_values = new_values * slope_correction + intercept_correction
                    return new_values

                return calibration_function_ba

            elif self.reference_method_type == "hard":
                # The calibration is done from a XY plot perspective
                intercept, slope, _ = wls_wrapper(x=self.ref_values_of_interest,
                                                  y=self.new_values_of_interest,
                                                  sample_weight=self.w_ref_values)

                def calibration_function_xy(new_values):
                    return (new_values - intercept) / slope

                return calibration_function_xy

    def calculate_error_metrics(self):
        if not self._has_error_metrics:
            possible = self._linreg_possible if self.reference_method_type == "hard" else self._ba_possible
            if possible:
                sample_weight = self.w_ref_values if self.reference_method_type == "hard" else self.w_mean_values
                self.metrics["mean_error"], self.metrics["std_error"] = \
                    weighted_error_mean_and_std(self.ref_values_of_interest,
                                                self.new_values_of_interest,
                                                sample_weight=sample_weight)
            else:
                self.metrics["mean_error"], self.metrics["std_error"] = \
                    np.nan, np.nan
            self._has_error_metrics = True

    def calculate_regression_metrics(self):
        if not self._has_regression_metrics:
            possible = self._linreg_possible if self.reference_method_type == "hard" else self._ba_possible
            if possible:
                sample_weight = self.w_ref_values if self.reference_method_type == "hard" else self.w_mean_values
                self.metrics["mae"] = mean_absolute_error(self.ref_values_of_interest,
                                                          self.new_values_of_interest,
                                                          sample_weight=sample_weight)
                self.metrics["mse"] = mean_squared_error(self.ref_values_of_interest,
                                                         self.new_values_of_interest,
                                                         sample_weight=sample_weight)
                self.metrics["rmse"] = np.sqrt(self.metrics["mse"])
                self.metrics["coef_of_det_r2"] = r2_score(self.ref_values_of_interest,
                                                          self.new_values_of_interest,
                                                          sample_weight=sample_weight)
            else:
                self.metrics["mae"] = np.nan
                self.metrics["mse"] = np.nan
                self.metrics["rmse"] = np.nan
                self.metrics["coef_of_det_r2"] = np.nan
            self._has_regression_metrics = True

    def calculate_ba_metrics(self):
        if self.reference_method_type == "hard":
            raise ValueError("Bland-Altman is not to be used with hard reference values (the whole point of "
                             "Bland-Altman is to account for variability in the reference values)")

        if self._ba_possible:
            # Calculate the trend:
            # Use WLS ("hard reference") as the X-axis is the mean of the two methods
            self.metrics["ba_intercept"], self.metrics["ba_slope"], self.metrics["ba_corr"] = \
                wls_wrapper(self.mean_values_of_interest,
                            self.diff_values_of_interest,
                            sample_weight=self.w_mean_values)

            # Calculate mean and std error
            self.calculate_error_metrics()
            self.metrics["ba_loa"] = self.limit_of_agreement * self.metrics["std_error"]

        else:
            self.metrics["ba_intercept"], self.metrics["ba_slope"], self.metrics["ba_loa"] = np.nan, np.nan, np.nan

    def bootstrap_error_metrics(self):
        if not self._has_error_metrics_bootstrap_ci:
            possible = self._linreg_possible if self.reference_method_type == "hard" else self._ba_possible
            if possible:
                # NOTE: The standard deviation (SD) lower and upper bound are not calculated. Bootstrapping the SD is
                # likely to (alpha) and (1-alpha) quantiles not being centered on the SD (pulled towards 0). Indeed, due
                # to duplication of certain samples in the random choice procedure, the SD is likely to be
                # underestimated. This is especially true for small sample sizes.
                sample_weight = self.w_ref_values if self.reference_method_type == "hard" else self.w_mean_values
                [self.metrics["mean_error"],
                 self.metrics["mean_error_low_bound"],
                 self.metrics["mean_error_up_bound"]], \
                [self.metrics["std_error"],
                 _,
                 _] = \
                    self.bootstrap_metric(ref_values=self.ref_values_of_interest,
                                          new_values=self.new_values_of_interest,
                                          sample_weight=sample_weight,
                                          func=weighted_error_mean_and_std
                                          )
            else:
                [self.metrics["mean_error"],
                 self.metrics["mean_error_low_bound"],
                 self.metrics["mean_error_up_bound"]], \
                [self.metrics["std_error"],
                 _,
                 _] = \
                    [np.nan, np.nan, np.nan], \
                    [np.nan, np.nan, np.nan]

            self._has_error_metrics_bootstrap_ci = True

    def bootstrap_regression_metrics(self):
        if not self._has_regression_metrics_bootstrap_ci:
            possible = self._linreg_possible if self.reference_method_type == "hard" else self._ba_possible
            if possible:
                sample_weight = self.w_ref_values if self.reference_method_type == "hard" else self.w_mean_values
                self.metrics["mae"], self.metrics["mae_low_bound"], self.metrics[
                    "mae_up_bound"] = self.bootstrap_metric(
                    ref_values=self.ref_values_of_interest,
                    new_values=self.new_values_of_interest,
                    sample_weight=sample_weight,
                    func=mean_absolute_error
                )
                self.metrics["mse"], self.metrics["mse_low_bound"], self.metrics[
                    "mse_up_bound"] = self.bootstrap_metric(
                    ref_values=self.ref_values_of_interest,
                    new_values=self.new_values_of_interest,
                    sample_weight=sample_weight,
                    func=mean_squared_error
                )
                self.metrics["rmse"], \
                self.metrics["rmse_low_bound"], \
                self.metrics["rmse_up_bound"] = np.sqrt(self.metrics["mse"]), \
                                                np.sqrt(self.metrics["mse_low_bound"]), \
                                                np.sqrt(self.metrics["mse_up_bound"])
                self.metrics["coef_of_det_r2"], self.metrics["coef_of_det_r2_low_bound"], \
                self.metrics["coef_of_det_r2_up_bound"] = self.bootstrap_metric(ref_values=self.ref_values_of_interest,
                                                                                new_values=self.new_values_of_interest,
                                                                                sample_weight=sample_weight,
                                                                                func=r2_score
                                                                                )
            else:
                self.metrics["mae"], self.metrics["mae_low_bound"], self.metrics[
                    "mae_up_bound"] = np.nan, np.nan, np.nan
                self.metrics["mse"], self.metrics["mse_low_bound"], self.metrics[
                    "mse_up_bound"] = np.nan, np.nan, np.nan
                self.metrics["rmse"], self.metrics["rmse_low_bound"], self.metrics[
                    "rmse_up_bound"] = np.nan, np.nan, np.nan
                self.metrics["coef_of_det_r2"], self.metrics["coef_of_det_r2_low_bound"], self.metrics[
                    "coef_of_det_r2_up_bound"] = np.nan, np.nan, np.nan
            self._has_regression_metrics_bootstrap_ci = True

    def reset(self):
        # Property attributes:
        self._mean_values = None
        self._diff_values = None
        self._w_ref_values = None
        self._w_new_values = None
        self._w_mean_values = None
        self._ref_values_of_interest = None
        self._new_values_of_interest = None
        self._mean_values_of_interest = None
        self._diff_values_of_interest = None
        self._calibration_function = None
        self._heteroscedasticity_info = None

        # Not property attributes
        self._ref_values_hist = None
        self._new_values_hist = None
        self._mean_values_hist = None
        self._ref_values_hist_smoothed = None
        self._new_values_hist_smoothed = None
        self._mean_values_hist_smoothed = None
        self._ref_values_hist_bin_edges = None
        self._new_values_hist_bin_edges = None
        self._mean_values_hist_bin_edges = None
        self._ref_values_rolling_mean_window = None
        self._new_values_rolling_mean_window = None
        self._mean_values_rolling_mean_window = None
        self._range_of_interest_ref_values = None
        self._range_of_interest_new_values = None
        self._range_of_interest_mean_values = None
        self._linreg_possible = True
        self._ba_possible = True
        self.metrics = {}
        self._has_error_metrics = False
        self._has_error_metrics_bootstrap_ci = False
        self._has_regression_metrics = False
        self._has_regression_metrics_bootstrap_ci = False
        self._has_linear_regression_calculated = False
        self._has_bland_altman_calculated = False
        self._h_residuals = None
        self._h_residuals_of_interest = None
        self._ref_values_bounds = None
        self._new_values_bounds = None
        self._mean_values_bounds = None

        self.set_data_range()

    def plot_regression(self,
                        xlim: list = [0, 1],
                        ylim: list = [0, 1],
                        title: Optional[str] = None,
                        dpi: int = 100,
                        alpha: float = 0.5,
                        plot_linreg: bool = True,
                        show_legend: bool = True,
                        ax: Optional[plt.Axes] = None,
                        xlabel: str = "X",
                        ylabel: str = "Y",
                        scatter_color: Optional[str] = None,
                        show_legend_title: bool = True,
                        **kwargs):
        """
        Plots a regression scatter plot comparing reference and new values, with options for linear regression fitting,
        legends, and customization of appearance.

        Parameters
        ----------
        xlim : list, optional, default: [0, 1]
            The x-axis limits for the plot as a list [xmin, xmax].

        ylim : list, optional, default: [0, 1]
            The y-axis limits for the plot as a list [ymin, ymax].

        title : str, optional, default: None
            The title of the plot. If None, no title is displayed.

        dpi : int, optional, default: 100
            The resolution of the plot in dots per inch.

        alpha : float, optional, default: 0.5
            The transparency level for scatter points. Values range from 0 (fully transparent) to 1 (fully opaque).

        plot_linreg : bool, optional, default: True
            Whether to plot a linear regression line over the scatter plot.

        show_legend : bool, optional, default: True
            Whether to show the legend in the plot.

        ax : matplotlib.axes._axes.Axes, optional, default: None
            An existing matplotlib Axes object to plot on. If None, a new figure and Axes object are created.

        xlabel : str, optional, default: "X"
            The label for the x-axis.

        ylabel : str, optional, default: "Y"
            The label for the y-axis.

        scatter_color : str, optional, default: None
            The color of the scatter points. If None, the default color cycle is used.

        show_legend_title : bool, optional, default: True
            Whether to display the title of the legend.

        **kwargs : dict
            Additional keyword arguments.
        """
        self.set_fonts(**kwargs)
        self.calculate_error_metrics()
        self.calculate_regression_metrics()
        self.fit_linear_regression()

        if ax is None:
            _, ax = plt.subplots(dpi=dpi, figsize=(8, 8))
        ax.set_aspect("equal")
        ax.plot(xlim, xlim, color="gray", alpha=0.5)
        units_suffix = ""
        if self.units is not None:
            units_suffix = self.units
        legend_title_metrics = ""
        legend_title_regression = ""
        if show_legend_title:
            legend_title_metrics = boldify_legend_text("Model metrics")
            legend_title_regression = boldify_legend_text("{0} regression".format(self.regression_method_name))
            if self.weighting != None:
                legend_title_metrics += "\n" + boldify_legend_text("{0}".format(self._weighting_method.capitalize()))
                legend_title_regression += "\n" + boldify_legend_text("{0}".format(self._weighting_method.capitalize()))
            legend_title_metrics += "\n"
            legend_title_regression += "\n"
        label_scatter = \
            legend_title_metrics \
            + "MAE: " + self.format_dimensional.format(self.metrics["mae"]) + units_suffix \
            + "\nMean error (SD): " + self.format_dimensional.format(self.metrics["mean_error"]) + units_suffix \
            + " (" + self.format_dimensional.format(self.metrics["std_error"]) + ")" \
            + "\nR$^2$ ($y$ vs. $x$): " + self.format_non_dimensional.format(self.metrics["coef_of_det_r2"])
        ax.scatter(self.ref_values, self.new_values,
                   color=scatter_color,
                   alpha=alpha,
                   label=label_scatter)

        if self._linreg_possible and plot_linreg:
            if self.range_of_interest is not None:
                if self.reference_method_type == "soft":
                    regline_x, _ = get_regression_line_soft_range_of_interest_intersections(
                        linreg_slope=self.metrics["linreg_slope"],
                        linreg_intercept=self.metrics["linreg_intercept"],
                        range_of_interest=self.range_of_interest)
                    if regline_x == []:
                        regline_x = xlim
                else:
                    regline_x = self.range_of_interest
            else:
                regline_x = xlim
            regline_y = [item * self.metrics["linreg_slope"] + self.metrics["linreg_intercept"] for item in regline_x]
            label_line = \
                legend_title_regression \
                + "Slope: " + self.format_non_dimensional.format(self.metrics["linreg_slope"]) \
                + "\nIntercept: " + self.format_dimensional.format(self.metrics["linreg_intercept"]) + units_suffix \
                + "\nPearson r ($y$ vs. $\hat{y}$): " + self.format_non_dimensional.format(
                    self.metrics["linreg_pearson_r"])
            ax.plot(regline_x,
                    regline_y,
                    color="black",
                    linestyle='--',
                    label=label_line)
        if self.reference_method_type == "hard" and self.range_of_interest:
            ax.axvspan(xmin=self.range_of_interest[0],
                       xmax=self.range_of_interest[1],
                       color="gray",
                       alpha=0.2,
                       label="Range of interest",
                       zorder=0)
        elif self.reference_method_type == "soft" and self.range_of_interest:
            center = 0.5 * (self.range_of_interest[0] + self.range_of_interest[1])
            half_diag = max(self.range_of_interest) - min(self.range_of_interest)
            """ax.fill_between(x=[center - half_diag, center, center + half_diag],
                             y1=[center, center - half_diag, center],
                             y2=[center, center + half_diag, center],
                             color="gray",
                             alpha=0.2,
                             label="Range of interest",
                             zorder=0
                             )"""
            ax.fill_between(x=[min(xlim), max(xlim)],
                            y1=[2 * center - half_diag - min(xlim), 2 * center - half_diag - max(xlim)],
                            y2=[2 * center + half_diag - min(xlim), 2 * center + half_diag - max(xlim)],
                            color="gray",
                            alpha=0.2,
                            label="Range of interest",
                            zorder=0
                            )

        if show_legend:
            ax.legend(facecolor="white", frameon=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_bland_altman(self,
                          xlim: list = [0, 1],
                          ylim: list = [0, 1],
                          title: Optional[str] = None,
                          dpi: int = 100,
                          alpha: float = 0.5,
                          ax: Optional[plt.Axes] = None,
                          plot_linreg: bool = True,
                          provide_slope: bool = True,
                          provide_correlation: bool = False,
                          show_legend: bool = True,
                          plot_trend_only: bool = False,
                          xlabel: str = "(Y + X) / 2",
                          ylabel: str = "Y - X",
                          scatter_color: Optional[str] = None,
                          regline_color: str = "m-",
                          text_bg_alpha: float = 0.5,
                          show_legend_title: bool = True,
                          **kwargs):
        """
        Plots a Bland-Altman plot to visualize the agreement between new and reference values. The plot shows the
        differences against the mean of new and reference values, with options for adding a regression
        trendline and additional statistics.

        Parameters
        ----------
        xlim : list, optional, default: [0, 1]
            The x-axis limits for the plot as a list [xmin, xmax].

        ylim : list, optional, default: [0, 1]
            The y-axis limits for the plot as a list [ymin, ymax].

        title : str, optional, default: None
            The title of the plot. If None, no title is displayed.

        dpi : int, optional, default: 100
            The resolution of the plot in dots per inch.

        alpha : float, optional, default: 0.5
            The transparency level for scatter points. Values range from 0 (fully transparent) to 1 (fully opaque).

        ax : matplotlib.axes._axes.Axes, optional, default: None
            An existing matplotlib Axes object to plot on. If None, a new figure and Axes object are created.

        plot_linreg : bool, optional, default: True
            Whether to plot a regression line (difference vs. mean) over the Bland-Altman plot.

        provide_slope : bool, optional, default: True
            Whether to annotate the plot with the slope of the regression line.

        provide_correlation : bool, optional, default: False
            Whether to annotate the plot with the correlation coefficient between the differences and mean.

        show_legend : bool, optional, default: True
            Whether to display a legend on the plot.

        plot_trend_only : bool, optional, default: False
            If True, only the regression trendline is plotted without scatter points.

        xlabel : str, optional, default: "(Y + X) / 2"
            The label for the x-axis.

        ylabel : str, optional, default: "Y - X"
            The label for the y-axis.

        scatter_color : str, optional, default: None
            The color of the scatter points. If None, the default color cycle is used.

        regline_color : str, optional, default: "m-"
            The color and style of the regression line (e.g., "m-" for a magenta solid line).

        text_bg_alpha : float, optional, default: 0.5
            The transparency level for the background of text annotations in the figure.

        show_legend_title : bool, optional, default: True
            Whether to include a title in the legend.

        **kwargs : dict
            Additional keyword arguments.
        """
        self.set_fonts(**kwargs)
        self.fit_bland_altman()

        meanColour = 'black'
        loaColour = 'black'

        if ax is None:
            _, ax = plt.subplots(dpi=dpi, figsize=(8, 5))
        ax.set_aspect("equal")
        if self.range_of_interest:
            xmin = (min(self.range_of_interest) - min(xlim)) / (max(xlim) - min(xlim))
            xmax = (max(self.range_of_interest) - min(xlim)) / (max(xlim) - min(xlim))
        else:
            xmin, xmax = 0, 1

        if plot_linreg:
            if self.range_of_interest:
                x_trend = [min(self.range_of_interest), max(self.range_of_interest)]
            else:
                x_trend = xlim
            y_trend = [self.metrics["ba_slope"] * x + self.metrics["ba_intercept"] for x in x_trend]
            slope_label = "$diff$ ~ " + self.format_non_dimensional.format(self.metrics["ba_slope"]) + " $\cdot$ $mean$"
            if self.confidence_interval is not None:
                slope_label += \
                    " (" + self.format_non_dimensional.format(self.metrics["ba_slope_low_bound"]) + " - " \
                    + self.format_non_dimensional.format(self.metrics["ba_slope_up_bound"]) + ")"
            correlation_label = ""
            if provide_correlation:
                correlation_label = "Pearson's r: " + self.format_non_dimensional.format(self.metrics["ba_corr"])
                if self.confidence_interval is not None:
                    correlation_label += \
                        " (" + self.format_non_dimensional.format(self.metrics["ba_corr_low_bound"]) + " - " \
                        + self.format_non_dimensional.format(self.metrics["ba_corr_up_bound"]) + ")"
            linreg_label = ""
            if provide_slope:
                linreg_label = slope_label
            if provide_correlation:
                if linreg_label == "":
                    linreg_label = correlation_label
                else:
                    linreg_label = "{}\n{}".format(linreg_label, correlation_label)
            ax.plot(x_trend,
                    y_trend,
                    regline_color,
                    label=linreg_label)

        if not plot_trend_only:
            ax.axhline(self.metrics["mean_error"], xmin=xmin, xmax=xmax,
                       color=meanColour, linestyle='--')
            ax.axhline(self.metrics["mean_error"] + self.limit_of_agreement * self.metrics["std_error"], xmin=xmin,
                       xmax=xmax,
                       color=loaColour, linestyle='dotted')
            ax.axhline(self.metrics["mean_error"] - self.limit_of_agreement * self.metrics["std_error"], xmin=xmin,
                       xmax=xmax,
                       color=loaColour, linestyle='dotted')

        if self.range_of_interest:
            ax.axvspan(xmin=self.range_of_interest[0],
                       xmax=self.range_of_interest[1],
                       color="gray",
                       alpha=0.2,
                       label="Range of interest",
                       zorder=0)

        ax.scatter(self.mean_values, self.diff_values, color=scatter_color, alpha=alpha)

        trans = transforms.blended_transform_factory(
            ax.transAxes, ax.transData)

        self.limit_of_agreement_range = (self.metrics["mean_error"] + (
                self.limit_of_agreement * self.metrics["std_error"])) - \
                                     (self.metrics["mean_error"] - self.limit_of_agreement * self.metrics["std_error"])
        offset = (self.limit_of_agreement_range / 100.0) * 1.5

        bbox_dict = dict(facecolor='white', edgecolor="none", pad=0, alpha=text_bg_alpha)

        if not plot_trend_only:
            if self._ba_possible:
                text_h_alignment = kwargs.get("text_h_alignment", "right")
                if text_h_alignment == "right":
                    x_text = xmax - 0.01
                else:
                    x_text = xmin + 0.01
                if min(ylim) <= self.metrics["mean_error"] <= max(ylim):
                    ax.text(x_text,
                            self.metrics["mean_error"] + offset,
                            'Mean',
                            ha=text_h_alignment, va="bottom",
                            transform=trans, bbox=bbox_dict)
                    ax.text(x_text,
                            self.metrics["mean_error"] - offset,
                            f'{self.metrics["mean_error"]:.2f}',
                            ha=text_h_alignment, va="top", transform=trans, bbox=bbox_dict)

                if min(ylim) <= self.metrics["mean_error"] + (self.limit_of_agreement * self.metrics["std_error"]) <= max(
                        ylim):
                    ax.text(x_text,
                            self.metrics["mean_error"] + (self.limit_of_agreement * self.metrics["std_error"]) + offset,
                            f'+{self.limit_of_agreement:.2f} SD',
                            ha=text_h_alignment, va="bottom",
                            transform=trans, bbox=bbox_dict)
                    ax.text(x_text,
                            self.metrics["mean_error"] + (self.limit_of_agreement * self.metrics["std_error"]) - offset,
                            f'{self.metrics["mean_error"] + self.limit_of_agreement * self.metrics["std_error"]:.2f}',
                            ha=text_h_alignment, va="top",
                            transform=trans, bbox=bbox_dict)

                if min(ylim) <= self.metrics["mean_error"] - (self.limit_of_agreement * self.metrics["std_error"]) <= max(
                        ylim):
                    ax.text(x_text,
                            self.metrics["mean_error"] - (self.limit_of_agreement * self.metrics["std_error"]) - offset,
                            f'-{self.limit_of_agreement:.2f} SD',
                            ha=text_h_alignment, va="top",
                            transform=trans, bbox=bbox_dict)
                    ax.text(x_text,
                            self.metrics["mean_error"] - (self.limit_of_agreement * self.metrics["std_error"]) + offset,
                            f'{self.metrics["mean_error"] - self.limit_of_agreement * self.metrics["std_error"]:.2f}',
                            ha=text_h_alignment, va="bottom",
                            transform=trans, bbox=bbox_dict)

        # Hide the right and top spines
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show_legend:
            legend_title = ""
            if show_legend_title:
                legend_title = boldify_legend_text("Bland") + boldify_legend_text("-") + \
                               boldify_legend_text("Altman analysis")
                if self.weighting is not None:
                    legend_title += "\n" + boldify_legend_text("with {0}".format(self._weighting_method))
            ax.legend(facecolor="white", frameon=True, title=legend_title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if title is not None:
            ax.set_title(title)
        return ax

    def plot_heteroscedasticity(self,
                                xlim: list = [0, 1],
                                ylim: Optional[list] = None,
                                title: Optional[str] = None,
                                dpi: int = 100,
                                alpha: float = 0.5,
                                show_legend: bool = True,
                                ax: Optional[plt.Axes] = None,
                                xlabel: Optional[str] = None,
                                ylabel: Optional[str] = None,
                                scatter_color: Optional[str] = None,
                                **kwargs):
        """
        Plots heteroscedasticity by visualizing the relationship between the reference values (in case of hard labels)
        or mean values (in case of soft reference values) and their residuals, allowing an assessment of whether the
        variance of residuals is constant across the range of values. Implements White and Breusch-Pagan tests. White
        test can detect cases where heteroscedasticity is non-monotonous.

        Parameters
        ----------
        xlim : list, optional, default: [0, 1]
            The x-axis limits for the plot as a list [xmin, xmax].

        ylim : list, optional, default: None
            The y-axis limits for the plot as a list [ymin, ymax]. If None, the limits
            are determined automatically based on the data.

        title : str, optional, default: None
            The title of the plot. If None, no title is displayed.

        dpi : int, optional, default: 100
            The resolution of the plot in dots per inch.

        alpha : float, optional, default: 0.5
            The transparency level for scatter points. Values range from 0 (fully transparent) to 1 (fully opaque).

        show_legend : bool, optional, default: True
            Whether to display a legend on the plot.

        ax : matplotlib.axes._axes.Axes, optional, default: None
            An existing matplotlib Axes object to plot on. If None, a new figure and Axes object are created.

        xlabel : str, optional, default: None
            The label for the x-axis. If None, a default label is applied based on the data being visualized.

        ylabel : str, optional, default: None
            The label for the y-axis. If None, "Absolute Residuals" is used.

        scatter_color : str, optional, default: None
            The color of the scatter points. If None, the default color cycle is used.

        **kwargs : dict
            Additional keyword arguments.
        """
        self.set_fonts(**kwargs)
        self.calculate_error_metrics()
        self.calculate_regression_metrics()
        self.fit_linear_regression()
        _ = self.heteroscedasticity_info

        if xlabel is None:
            if self.reference_method_type == "hard":
                xlabel = "X"
            else:
                xlabel = "(Y + X) / 2"
        if ylabel is None:
            ylabel = "Absolute residuals"

        if ax is None:
            _, ax = plt.subplots(dpi=dpi, figsize=(8, 8))
        y_values = np.abs(self._h_residuals)
        y_values_of_interest = np.abs(self._h_residuals_of_interest)
        if self.reference_method_type == "hard":
            x_values = self.ref_values
            x_values_of_interest = self.ref_values_of_interest
        elif self.reference_method_type == "soft":
            x_values = self.mean_values
            x_values_of_interest = self.mean_values_of_interest

        ax.scatter(x_values,
                   y_values,
                   color=scatter_color,
                   alpha=alpha)

        if self.range_of_interest:
            ax.axvspan(xmin=self.range_of_interest[0],
                       xmax=self.range_of_interest[1],
                       color="gray",
                       alpha=0.2,
                       label="Range of interest",
                       zorder=0)
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([0, max(y_values) * 1.3])
        heteroscedasticity_tests = {"white": "White (non-monotonous)",
                                    "breuschpagan": "Breusch-Pagan (monotonous)"}
        heteroscedasticity_text = boldify_legend_text("Heteroscedasticity tests") + "\n"
        for i, type in enumerate(heteroscedasticity_tests.keys()):
            if self.heteroscedasticity_info[type]:
                heteroscedasticity_text += "{}: p<{:.02f}".format(heteroscedasticity_tests[type],
                                                                  self.significance_level)
            else:
                heteroscedasticity_text += "{}: ns. (p={:.04f})".format(heteroscedasticity_tests[type],
                                                                        self.heteroscedasticity_info["p_value_" + type])
            if i < len(heteroscedasticity_tests) - 1:
                heteroscedasticity_text += "\n"

        ax.text(0.02, 0.98, heteroscedasticity_text,
                transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="left", fontsize=12,
                bbox=dict(facecolor='white', pad=3))

        # Plot the envelope of the residuals
        n_bins = 20
        bins = np.linspace(min(x_values_of_interest), max(x_values_of_interest) + 1e-4, n_bins + 1)
        x_envelope = []
        y_envelopes = {0.6: [],
                       0.9: [],
                       0.95: []
                       }
        for i in range(n_bins):
            x_values_of_interest_quantile = x_values_of_interest[(x_values_of_interest >= bins[i]) &
                                                                 (x_values_of_interest < bins[i + 1])]
            y_values_of_interest_quantile = y_values_of_interest[(x_values_of_interest >= bins[i]) &
                                                                 (x_values_of_interest < bins[i + 1])]
            if len(x_values_of_interest_quantile) < self.min_samples_by_bin:
                warnings.warn("Heterocedasticity plot: there was less than {} samples in the bin, not plotting the "
                              "quantiles.")
                continue
            x_envelope.append(np.mean(x_values_of_interest_quantile))
            for quantile_y in y_envelopes.keys():
                y_envelopes[quantile_y].append(np.quantile(y_values_of_interest_quantile, q=quantile_y))
        x_envelope = np.array(x_envelope)
        for i, (quantile_y, y_envelope) in enumerate(y_envelopes.items()):
            y_envelope = np.array(y_envelope)
            ax.plot(x_envelope, y_envelope,
                    label="{:.00f}% of samples".format(quantile_y * 100),
                    color=sns.color_palette()[i + 1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_legend:
            ax.legend(facecolor="white", frameon=True, **kwargs)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_data_distribution(self, dpi=100, return_fig=False, title="", xlim=None, **kwargs):
        self.set_fonts(**kwargs)
        if self.reference_method_type == "hard":
            if self.weighting == "inverse":
                _ = self.w_ref_values
            nrows = 1
            names = ["reference values"]
            xs = [self.ref_values]
            original_hists = [self._ref_values_hist]
            smoothed_hists = [self._ref_values_hist_smoothed if self._linreg_possible else None]
            rolling_mean_windows = [self._ref_values_rolling_mean_window]
            bin_edgess = [self._ref_values_hist_bin_edges]
            ranges_of_interest = [self._range_of_interest_ref_values]
            display_range_of_interest = [True if self._linreg_possible else False]

        if self.reference_method_type == "soft":
            if self.weighting == "inverse":
                _ = self.w_ref_values
                _ = self.w_new_values
                _ = self.w_mean_values
            nrows = 3
            # names = ["reference values", "new values", "mean values"]
            # xs = [self.ref_values, self.new_values, self.mean_values]
            names = ["reference values of interest", "new values of interest", "mean values"]
            xs = [self.ref_values_of_interest, self.new_values_of_interest, self.mean_values]
            original_hists = [self._ref_values_hist, self._new_values_hist, self._mean_values_hist]
            smoothed_hists = [self._ref_values_hist_smoothed if self._linreg_possible else None,
                              self._new_values_hist_smoothed if self._linreg_possible else None,
                              self._mean_values_hist_smoothed if self._ba_possible else None]
            rolling_mean_windows = [self._ref_values_rolling_mean_window,
                                    self._new_values_rolling_mean_window,
                                    self._mean_values_rolling_mean_window]
            bin_edgess = [self._ref_values_hist_bin_edges,
                          self._new_values_hist_bin_edges,
                          self._mean_values_hist_bin_edges]
            ranges_of_interest = [self._range_of_interest_ref_values, self._range_of_interest_new_values, self.range_of_interest]
            display_range_of_interest = [True if self._linreg_possible else False,
                                         True if self._linreg_possible else False,
                                         True if self._ba_possible else False]

        fig, ax = plt.subplots(dpi=dpi, ncols=1, nrows=nrows, figsize=(7, nrows * 3.5 + 1))
        if nrows == 1:
            ax = [ax]

        for i, (name, x, original_hist, bin_edges, smoothed_hist, rolling_mean_window, roi, droi) in \
                enumerate(zip(names, xs, original_hists, bin_edgess, smoothed_hists, rolling_mean_windows,
                              ranges_of_interest, display_range_of_interest)):
            if self.weighting is None or len(self.ref_values_of_interest) == 0:
                sns.histplot(x=x, binwidth=self.binwidth, label="Distribution of " + name, ax=ax[i])
            elif self.weighting == "inverse":
                sns.histplot(x=x, bins=bin_edges, label="Distribution of " + name, ax=ax[i])
            if self.range_of_interest and droi:
                ax[i].axvspan(xmin=min(roi),
                              xmax=max(roi),
                              color="gray",
                              alpha=0.2,
                              label="Range of interest",
                              zorder=0)
            ax[i].set_xlabel(name.capitalize())

            if smoothed_hist is not None:
                ax[i].plot(bin_edges + self.binwidth / 2, smoothed_hist,
                           label="Moving average - Window: {:.02f}".format(rolling_mean_window))
                bin_centers = bin_edges + self.binwidth / 2
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    weighted_hist = original_hist * (1 / smoothed_hist) * np.max(smoothed_hist)
                if self.range_of_interest is not None:
                    indexes = (bin_centers >= min(roi)) & (bin_centers <= max(roi))
                    bin_centers = bin_centers[indexes]
                    weighted_hist = weighted_hist[indexes]
                ax[i].plot(bin_centers, weighted_hist, label="Inverse sampling reconstructed")


            if len(x) != 0:
                ax[i].legend(facecolor="white", frameon=True)
            if xlim:
                ax[i].set_xlim(xlim)

        plt.suptitle(title)

        if return_fig:
            return fig
        else:
            plt.show()
            plt.close()

    def set_style(self, style=DEFAULT_STYLE):
        """
        Sets the style for Matplotlib plots using the specified style configuration.

        Parameters
        ----------
        style : str or dict, optional, default: DEFAULT_STYLE from the config.py
            The style to apply to Matplotlib plots. Can be:
            - A predefined Matplotlib style string (e.g., "ggplot", "seaborn").
            - A dictionary of rcParams for custom styling.
            - The `DEFAULT_STYLE` constant for a default configuration.

        Notes
        -----
        - The current rcParams are stored before applying the new style,
          allowing a reset later using the `reset_style` method.
        - This method modifies global Matplotlib settings.
        """
        self.prev_rc_params = plt.rcParams.copy()
        plt.style.use(style)

    def reset_style(self):
        """
        Resets the Matplotlib style to the settings stored before calling `set_style`.

        Notes
        -----
        - The rcParams are restored to the copy stored in the `prev_rc_params` attribute.
        - If `set_style` has not been called, this method does nothing.
        """
        plt.rcParams.update(self.prev_rc_params)

    @staticmethod
    def set_fonts(**kwargs):
        rc_params_mapping = {
            'xticks_fontsize': ('xtick', 'labelsize'),
            'yticks_fontsize': ('ytick', 'labelsize'),
            'legend_fontsize': ('legend', 'fontsize'),
            'title_fontsize': ('axes', 'titlesize'),
            'label_size': ('axes', 'labelsize'),
            'text_fontsize': ('font', 'size'),  # remaining text
        }

        for key, (rc_section, rc_param) in rc_params_mapping.items():
            if key in kwargs:
                plt.rc(rc_section, **{rc_param: kwargs[key]})


def wls_wrapper(x, y, sample_weight, return_std: bool = False):
    model = sm.WLS(y, sm.add_constant(x), weights=1. if sample_weight is None else sample_weight).fit()
    if len(model.params) != 2:
        if return_std:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan
    else:
        intercept = model.params[0]
        slope = model.params[1]
        pearson_corr = DescrStatsW(data=np.array([x, y]).T, weights=sample_weight).corrcoef[0][1]
        if return_std:
            intercept_std = model.bse[0]
            slope_std = model.bse[1]
            return intercept, slope, pearson_corr, intercept_std, slope_std
        else:
            return intercept, slope, pearson_corr


def wlp_wrapper(x, y, sample_weight_x, sample_weight_y):
    # NOTE: It is unclear how the weighting should be handled for Least Products regression.
    # Here the weights of the reference values and new values are used for the two Least Square regressions, but
    # perhaps an alternative is to use the weights of the means.
    _, slope_a, _ = wls_wrapper(x, y, sample_weight=sample_weight_x)
    _, slope_b, _ = wls_wrapper(y, x, sample_weight=sample_weight_y)
    slope_b = 1 / slope_b
    if np.sign(slope_a) != np.sign(slope_b):
        warnings.warn('Type I regressions of opposite sign. Do not trust least products regression slope')
        return np.array([np.nan, np.nan, np.nan])
    slope = np.sign(slope_a) * np.sqrt(slope_a * slope_b)
    if sample_weight_x is None and sample_weight_y is None:
        intercept = np.mean(y) - slope * np.mean(x)
    else:
        intercept = np.median(y) - slope * np.median(x)
    pearson_corr = slope_a / slope
    return intercept, slope, pearson_corr
