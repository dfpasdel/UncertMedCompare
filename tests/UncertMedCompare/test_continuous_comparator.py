import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from unittest import TestCase

from UncertMedCompare.continuous_comparator import ContinuousComparator
from UncertMedCompare.config import DEFAULT_STYLE

plt.style.use(DEFAULT_STYLE)


class TestContinuousComparator(TestCase):
    def setUp(self):
        self.timeout_plots = 1500  # Set None/False if you want to avoid closing plots
        self.show_fig = partial(self._show_fig, timeout=self.timeout_plots)

    def tearDown(self):
        plt.close()

    #########################################################
    #        Histogram smoothing tests (weighting)          #
    #########################################################

    def test_sparse_histogram_smoothing(self):
        x_a = np.random.normal(0.30, 0.01, 100)
        x_b = np.random.normal(0.70, 0.01, 300)
        x = np.concatenate([x_a, x_b])

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=x,
                                          range_of_interest=[0.2, 0.8],
                                          binwidth=0.01,
                                          weighting="inverse")

        fig = comparator.plot_data_distribution(xlim=[0, 1], return_fig=True)
        self.show_fig(fig, title=self._testMethodName)

        assert (min(comparator._ref_values_hist) == 0)
        assert (min(comparator._ref_values_hist_smoothed) > 0)

    def test_sparse_histogram_smoothing_out_of_range_values(self):
        x_a = np.random.normal(0.30, 0.07, 100)
        x_b = np.random.normal(0.70, 0.07, 100)
        x = np.concatenate([x_a, x_b])

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=x,
                                          range_of_interest=[0.2, 0.8],
                                          binwidth=0.01,
                                          weighting="inverse")

        fig = comparator.plot_data_distribution(xlim=[0, 1], return_fig=True)
        self.show_fig(fig, title=self._testMethodName)

        assert (min(comparator._ref_values_hist_smoothed) > 0)

    #########################################################
    #           Least squares regression tests              #
    #########################################################

    def test_linear_regression_ls(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 0.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          reference_method_type="hard")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100], ylim=[0, 100], ax=ax)
        self.show_fig(fig=fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    def test_linear_regression_ls_with_range(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 0.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          range_of_interest=[30, 70],
                                          reference_method_type="hard")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    def test_linear_regression_with_bias_ls(self):
        mean = 50
        x_gt = np.random.normal(mean, 5, 100000)
        std_x = 0.
        std_y = 2.
        fbias = 10.
        pbias = 1.2
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt * pbias + fbias + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[20, 80],
                                          binwidth=1,
                                          reference_method_type="hard")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_intercept"], fbias, atol=0.1 * mean, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_slope"], pbias, atol=0., rtol=0.05))

    def test_linear_regression_ls_constant(self):
        x_gt = np.ones(10000)
        x = x_gt
        y = x_gt

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          reference_method_type="hard")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100], ylim=[0, 100], ax=ax)
        self.show_fig(fig, title=self._testMethodName)

    def test_linear_regression_inverse_weighting_ls(self):
        weighting = "inverse"

        mean_a = 30
        x_gt_a = np.random.normal(mean_a, 7, 100000)
        mean_b = 70
        x_gt_b = np.random.normal(mean_b, 7, 100000)
        std_x = 0.
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

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[20, 80],
                                          binwidth=1,
                                          reference_method_type="hard",
                                          weighting=weighting)

        # Alternate samples with different distribution of the label values
        x_undersampled = np.concatenate([x_a[:2000], x_b])
        y_undersampled = np.concatenate([y_a[:2000], y_b])
        weighted_comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                                new_method_measurements=y_undersampled,
                                                                range_of_interest=[20, 80],
                                                                binwidth=1,
                                                                reference_method_type="hard",
                                                                weighting=weighting)

        fig = comparator.plot_data_distribution(xlim=[0, 100], return_fig=True)
        self.show_fig(fig, title=self._testMethodName + "_Original_data_distribution")
        fig = weighted_comparator_undersampled.plot_data_distribution(xlim=[0, 100], return_fig=True)
        self.show_fig(fig, title=self._testMethodName + "_Under_sampled_data_distribution")
        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        comparator.plot_regression(xlim=[0, 100], ylim=[0, 100], ax=axs[0], title="Original sampling")
        weighted_comparator_undersampled.plot_regression(xlim=[0, 100], ylim=[0, 100], ax=axs[1],
                                                         title="Under sampling")
        self.show_fig(fig, title=self._testMethodName + "_" + str(weighting) + "_weighting")

        assert (np.isclose(weighted_comparator_undersampled.metrics["linreg_slope"],
                           comparator.metrics["linreg_slope"],
                           atol=0., rtol=0.02))
        assert (np.isclose(weighted_comparator_undersampled.metrics["linreg_intercept"],
                           comparator.metrics["linreg_intercept"],
                           atol=0.05 * np.mean([mean_a, mean_b]), rtol=0.))

    #########################################################
    #           Least products regression tests             #
    #########################################################

    def test_linear_regression_lp(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 2.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          reference_method_type="soft")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    def test_linear_regression_lp_with_range(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 2.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          range_of_interest=[30, 70],
                                          reference_method_type="soft")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    def test_linear_regression_lp_inverse_weighting_no_range(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 2.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          reference_method_type="soft",
                                          weighting="inverse")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))
        assert (-1 <= comparator.metrics["linreg_pearson_r"] <= 1)

    def test_linear_regression_lp_inverse_weighting(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 2.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[20, 80],
                                          binwidth=1,
                                          reference_method_type="soft",
                                          weighting="inverse")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    def test_linear_regression_with_bias_lp(self):
        mean = 50
        x_gt = np.random.normal(mean, 10, 100000)
        std_x = 3.
        std_y = 3.
        fixed_bias = 15
        proportional_bias = 0.7
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt * proportional_bias + fixed_bias + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[20, 80],
                                          binwidth=1,
                                          reference_method_type="soft")

        fig = comparator.plot_data_distribution(xlim=[0, 100], return_fig=True)
        self.show_fig(fig, title=self._testMethodName)
        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        # NOTE: The slope and intercept do not need to correspond to the one specified in the biases above, because the
        # variability "leaks" into the slope. This could potentially be solved by using Least Perpendicular squares
        # (Major Axis) regression instead of Least Products (Reduced Major Axis) regression. See Ludbrook's paper:
        # (https://doi.org/10.1111/j.1440-1681.2010.05376.x)
        # However, a weighted version of the Least Perpendicular squares is not straightforward to implement.

    def test_linear_regression_inverse_weighting_lp(self):
        mean_a = 30
        x_gt_a = np.random.normal(mean_a, 7, 100000)
        mean_b = 70
        x_gt_b = np.random.normal(mean_b, 7, 100000)
        std_x = 4.
        std_y = 4.

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

        # Alternative samples with different distribution of the label values
        # NOTE: Should it be different distribution of the mean values instead?
        x_undersampled = np.concatenate([x_a[:2000], x_b])
        y_undersampled = np.concatenate([y_a[:2000], y_b])

        # NOTE: Inverse weighting implementation does not give the exact result for soft values.
        # The present test compare the effectiveness of weighting at making the metrics less data dependent
        weighting_methods = [None, "inverse"]
        metrics_ratios_by_weighting_methods = {}

        for weighting in weighting_methods:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              range_of_interest=[20, 80],
                                              binwidth=1,
                                              reference_method_type="soft",
                                              weighting=weighting)
            comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                           new_method_measurements=y_undersampled,
                                                           range_of_interest=[20, 80],
                                                           binwidth=1,
                                                           reference_method_type="soft",
                                                           weighting=weighting)

            fig = comparator.plot_data_distribution(xlim=[0, 100], return_fig=True)
            self.show_fig(fig, title=self._testMethodName + "_Original_data_distribution")
            fig = comparator_undersampled.plot_data_distribution(xlim=[0, 100], return_fig=True)
            self.show_fig(fig, title=self._testMethodName + "_Under_sampled_data_distribution")
            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            comparator.plot_regression(xlim=[0, 100],
                                       ylim=[0, 100],
                                       ax=axs[0],
                                       title="Original sampling")
            comparator_undersampled.plot_regression(xlim=[0, 100],
                                                    ylim=[0, 100],
                                                    ax=axs[1],
                                                    title="Under sampling")
            self.show_fig(fig, title=self._testMethodName + "_" + str(weighting) + "_weighting")

            metrics = comparator.metrics
            metrics_undersampled = comparator_undersampled.metrics
            metric_ratios = {}
            for k, v in metrics.items():
                metric_ratios[k] = metrics[k] / metrics_undersampled[k]
            metrics_ratios_by_weighting_methods[weighting] = metric_ratios

        for metric in metrics_ratios_by_weighting_methods[weighting_methods[0]].keys():
            if "_bound" in metric:
                # Do not check the confidence intervals, only the point estimates
                continue
            ratio_no_weighting = metrics_ratios_by_weighting_methods[None][metric]
            ratio_inverse_weighting = metrics_ratios_by_weighting_methods["inverse"][metric]
            # print(metric, ratio_no_weighting, ratio_inverse_weighting)
            assert (abs(ratio_inverse_weighting - 1) <= abs(ratio_no_weighting - 1))

    #########################################################
    #            Bland-Altman regression tests              #
    #########################################################

    def test_linear_regression_ba(self):
        x_gt = np.random.normal(50, 10, 100000)
        std_x = 2.
        std_y = 2.
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          reference_method_type="soft",
                                          soft_regression_method="BA")

        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator.metrics["linreg_slope"], 1, atol=0.01, rtol=0.))
        assert (np.isclose(comparator.metrics["linreg_intercept"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    #########################################################
    #             Bland-Altman analysis tests               #
    #########################################################

    def test_bland_altman_weighting(self):
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

        # NOTE: Inverse weighting implementation does not give the exact result for soft values.
        # The present test compare the effectiveness of weighting at making the metrics less data dependent
        weighting_methods = [None, "inverse"]
        metrics_ratios_by_weighting_methods = {}

        for weighting in weighting_methods:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              range_of_interest=[30, 70],
                                              binwidth=1,
                                              reference_method_type="soft",
                                              weighting=weighting,
                                              bootstrap_samples=1000)
            comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                           new_method_measurements=y_undersampled,
                                                           range_of_interest=[30, 70],
                                                           binwidth=1,
                                                           reference_method_type="soft",
                                                           weighting=weighting,
                                                           bootstrap_samples=1000)

            fig = comparator.plot_data_distribution(xlim=[0, 100], return_fig=True)
            self.show_fig(fig, title=self._testMethodName + "_original_sampling")
            fig = comparator_undersampled.plot_data_distribution(xlim=[0, 100], return_fig=True)
            self.show_fig(fig, title=self._testMethodName + "_unbalanced_sampling")

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            comparator.plot_bland_altman(xlim=[0, 100],
                                         ylim=[-40, 40],
                                         ax=axs[0],
                                         title="Original sampling")
            comparator_undersampled.plot_bland_altman(xlim=[0, 100],
                                                      ylim=[-40, 40],
                                                      ax=axs[1],
                                                      title="Unbalanced sampling")
            self.show_fig(fig, title=self._testMethodName + "_" + str(weighting) + "_weighting")

            metrics = comparator.metrics
            metrics_undersampled = comparator_undersampled.metrics
            metric_ratios = {}
            for k, v in metrics.items():
                metric_ratios[k] = metrics[k] / metrics_undersampled[k]
            metrics_ratios_by_weighting_methods[weighting] = metric_ratios

        for metric in metrics_ratios_by_weighting_methods[weighting_methods[0]].keys():
            if "_bound" in metric:
                # Do not check the confidence intervals, only the point estimates
                continue
            ratio_no_weighting = metrics_ratios_by_weighting_methods[None][metric]
            ratio_inverse_weighting = metrics_ratios_by_weighting_methods["inverse"][metric]
            print(metric, ratio_no_weighting, ratio_inverse_weighting)
            assert (abs(ratio_inverse_weighting - 1) <= abs(ratio_no_weighting - 1))

    #########################################################
    #                  Calibration tests                    #
    #########################################################

    def test_calibration_function_hard_reference(self):
        x_gt = np.random.normal(50, 5, 100000)
        std_x = 0.
        std_y = 2.
        pbias = 1.2
        fbias = 10
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt * pbias + fbias + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[20, 80],
                                          binwidth=1,
                                          reference_method_type="hard")
        calibration_function = comparator.calibration_function
        calibrated_y = calibration_function(y)
        comparator_calibrated = ContinuousComparator(reference_method_measurements=x,
                                                     new_method_measurements=calibrated_y,
                                                     range_of_interest=[20, 80],
                                                     binwidth=1,
                                                     reference_method_type="hard")

        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=axs[0],
                                   title="Non calibrated")
        comparator_calibrated.plot_regression(xlim=[0, 100],
                                              ylim=[0, 100],
                                              ax=axs[1],
                                              title="Calibrated")
        self.show_fig(fig, title=self._testMethodName)

        assert (np.isclose(comparator_calibrated.metrics["linreg_slope"],
                           1, atol=0.0001, rtol=0.))
        assert (np.isclose(comparator_calibrated.metrics["linreg_intercept"],
                           0, atol=0.0001 * (max(y) - min(y)), rtol=0.))

    def test_calibration_function_soft_reference(self):
        x_gt = np.random.uniform(20, 80, 100)
        std_x = 3.
        std_y = 3.
        pbias = 1.4
        fbias = 10
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt * pbias + fbias + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[30, 70],
                                          binwidth=1,
                                          reference_method_type="soft",
                                          confidence_interval=95)
        calibration_function = comparator.calibration_function
        calibrated_y = calibration_function(y)
        comparator_calibrated = ContinuousComparator(reference_method_measurements=x,
                                                     new_method_measurements=calibrated_y,
                                                     range_of_interest=[30, 70],
                                                     binwidth=1,
                                                     reference_method_type="soft",
                                                     confidence_interval=95)

        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        comparator.plot_bland_altman(xlim=[0, 100],
                                     ylim=[-40, 40],
                                     ax=axs[0],
                                     title="Non calibrated")
        comparator_calibrated.plot_bland_altman(xlim=[0, 100],
                                                ylim=[-40, 40],
                                                ax=axs[1],
                                                title="Calibrated")
        self.show_fig(fig, title=self._testMethodName)

        assert (comparator_calibrated.metrics["ba_slope"] - comparator_calibrated.metrics["ba_slope_up_bound"] <= 0)
        assert (comparator_calibrated.metrics["ba_slope"] - comparator_calibrated.metrics["ba_slope_low_bound"] >= 0)
        assert (np.isclose(comparator_calibrated.metrics["mean_error"], 0, atol=0.01 * (max(y) - min(y)), rtol=0.))

    #########################################################
    #             Regression metrics tests                  #
    #########################################################

    def test_regression_metrics_inverse_weighting(self):
        weighting = "inverse"

        x_gt_a = np.random.uniform(low=10, high=50, size=10000)
        x_gt_b = np.random.uniform(low=50, high=90, size=10000)

        # Simulate a linear model with heteroscedasticity
        def f(x):
            return 0.5 * x + 35 + np.random.normal(0, 1, len(x)) * (0.6 + abs(x - 70) / 7)

        x_a = x_gt_a
        x_b = x_gt_b
        y_a = f(x_gt_a)
        y_b = f(x_gt_b)

        # Original samples
        x = np.concatenate([x_a, x_b])
        y = np.concatenate([y_a, y_b])

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          binwidth=1,
                                          reference_method_type="hard",
                                          weighting=weighting)

        # Alternate samples with different distribution of the label values
        x_undersampled = np.concatenate([x_a[:1000], x_b])
        y_undersampled = np.concatenate([y_a[:1000], y_b])
        weighted_comparator_undersampled = ContinuousComparator(reference_method_measurements=x_undersampled,
                                                                new_method_measurements=y_undersampled,
                                                                binwidth=1,
                                                                reference_method_type="hard",
                                                                weighting=weighting)

        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        comparator.plot_regression(xlim=[0, 100],
                                   ylim=[0, 100],
                                   ax=axs[0],
                                   title="Original sampling")
        weighted_comparator_undersampled.plot_regression(xlim=[0, 100],
                                                         ylim=[0, 100],
                                                         ax=axs[1],
                                                         title="Under sampling")
        self.show_fig(fig, title=self._testMethodName + "_" + str(weighting) + "_weighting")

        assert (np.isclose(comparator.metrics["mae"],
                           weighted_comparator_undersampled.metrics["mae"], rtol=0.1))
        assert (np.isclose(comparator.metrics["mean_error"],
                           weighted_comparator_undersampled.metrics["mean_error"], rtol=0.1))
        assert (np.isclose(comparator.metrics["std_error"],
                           weighted_comparator_undersampled.metrics["std_error"], rtol=0.1))
        assert (np.isclose(comparator.metrics["coef_of_det_r2"],
                           weighted_comparator_undersampled.metrics["coef_of_det_r2"], rtol=0.1))
        assert (np.isclose(comparator.metrics["rmse"],
                           weighted_comparator_undersampled.metrics["rmse"], rtol=0.1))

    #########################################################
    #               Heteroscedasticity tests                #
    #########################################################

    def test_heteroscedasticity_false_hard_label(self):
        x_gt = np.random.normal(0.5, 0.15, 50000)
        x = x_gt
        y = x_gt + np.random.normal(0, 0.01, len(x_gt))

        for range_of_interest in [None, [0.4, 0.6]]:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="hard",
                                              range_of_interest=range_of_interest)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            heteroscedasticity_info = comparator.heteroscedasticity_info
            comparator.plot_regression(ax=axs[0])
            comparator.plot_heteroscedasticity(ax=axs[1])
            self.show_fig(fig=fig, title=self._testMethodName + " | Range: " + str(range_of_interest))
            assert (heteroscedasticity_info["white"] is False)
            assert (heteroscedasticity_info["breuschpagan"] is False)

    def test_heteroscedasticity_true_hard_label_monotonous(self):
        x_gt = np.random.normal(0.5, 0.15, 50000)
        x = x_gt
        y = []
        for i in range(len(x_gt)):
            y.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * x_gt[i])))
        y = np.array(y)

        for range_of_interest in [None, [0.4, 0.6]]:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="hard",
                                              range_of_interest=range_of_interest)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            heteroscedasticity_info = comparator.heteroscedasticity_info
            comparator.plot_regression(ax=axs[0])
            comparator.plot_heteroscedasticity(ax=axs[1])
            self.show_fig(fig=fig, title=self._testMethodName + " | Range: " + str(range_of_interest))
            assert (heteroscedasticity_info["white"] is True)
            assert (heteroscedasticity_info["breuschpagan"] is True)

    def test_heteroscedasticity_true_hard_label_non_monotonous(self):
        x_gt = np.random.normal(0.5, 0.15, 50000)
        x = x_gt
        y = []
        for i in range(len(x_gt)):
            y.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * (0.5 - abs(0.5 - x_gt[i])))))
        y = np.array(y)

        for range_of_interest in [None, [0.4, 0.6]]:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="hard",
                                              range_of_interest=range_of_interest)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            heteroscedasticity_info = comparator.heteroscedasticity_info
            comparator.plot_regression(ax=axs[0])
            comparator.plot_heteroscedasticity(ax=axs[1])
            self.show_fig(fig=fig, title=self._testMethodName + " | Range: " + str(range_of_interest))
            assert (heteroscedasticity_info["white"] is True)
            # NOTE: The Breusch-Pagan test does not necessarily detects when heteroscedasticity is monotonous

    def test_heteroscedasticity_false_soft_label(self):
        x_gt = np.random.normal(0.5, 0.15, 50000)
        x = x_gt + np.random.normal(0, 0.01, len(x_gt))
        y = x_gt + np.random.normal(0, 0.01, len(x_gt))

        for range_of_interest in [None, [0.4, 0.6]]:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="soft",
                                              range_of_interest=range_of_interest)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            heteroscedasticity_info = comparator.heteroscedasticity_info
            comparator.plot_bland_altman(xlim=[0, 1], ylim=[-0.5, 0.5], ax=axs[0])
            comparator.plot_heteroscedasticity(ax=axs[1])
            self.show_fig(fig=fig, title=self._testMethodName + " | Range: " + str(range_of_interest))
            assert (heteroscedasticity_info["breuschpagan"] is False)
            # NOTE: There is a side effect with soft values that can make the White test detect heteroscedasticity as if
            # the heteroscedasticity was not monotonous (see plot). Additional qualitative assessment of
            # heteroscedasticity is necessary.

    def test_heteroscedasticity_true_soft_label_monotonous(self):
        x_gt = np.random.normal(0.5, 0.15, 50000)
        x = []
        y = []
        for i in range(len(x_gt)):
            x.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * x_gt[i])))
            y.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * x_gt[i])))
        x = np.array(x)
        y = np.array(y)

        for range_of_interest in [None, [0.4, 0.6]]:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="soft",
                                              range_of_interest=range_of_interest)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            heteroscedasticity_info = comparator.heteroscedasticity_info
            comparator.plot_bland_altman(xlim=[0, 1], ylim=[-0.5, 0.5], ax=axs[0])
            comparator.plot_heteroscedasticity(ax=axs[1])
            self.show_fig(fig=fig, title=self._testMethodName + " | Range: " + str(range_of_interest))
            assert (heteroscedasticity_info["white"] is True)
            assert (heteroscedasticity_info["breuschpagan"] is True)

    def test_heteroscedasticity_true_soft_label_non_monotonous(self):
        x_gt = np.random.normal(0.5, 0.15, 50000)
        x = []
        y = []
        for i in range(len(x_gt)):
            x.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * (0.5 - abs(0.5 - x_gt[i])))))
            y.append(x_gt[i] + np.random.normal(0, max(0, 0.02 + 0.05 * (0.5 - abs(0.5 - x_gt[i])))))
        x = np.array(x)
        y = np.array(y)

        for range_of_interest in [None, [0.4, 0.6]]:
            comparator = ContinuousComparator(reference_method_measurements=x,
                                              new_method_measurements=y,
                                              reference_method_type="soft",
                                              range_of_interest=range_of_interest)

            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            heteroscedasticity_info = comparator.heteroscedasticity_info
            comparator.plot_bland_altman(xlim=[0, 1], ylim=[-0.5, 0.5], ax=axs[0])
            comparator.plot_heteroscedasticity(ax=axs[1])
            self.show_fig(fig=fig, title=self._testMethodName + " | Range: " + str(range_of_interest))
            assert (heteroscedasticity_info["white"] is True)
            # NOTE: The Breusch-Pagan test does not necessarily detects when heteroscedasticity is monotonous

    #########################################################
    #                    Diverse tests                      #
    #########################################################

    def test_correlation_inverse_weighting_ba(self):
        weighting = "inverse"

        x_gt_a = np.random.uniform(0.2, 0.8, 1000)
        x_gt_b = np.random.uniform(0.45, 0.55, 10000)
        std = 0.03

        x_a = x_gt_a + np.random.normal(0, std, len(x_gt_a))
        x_b = x_gt_b + np.random.normal(0, std, len(x_gt_b))
        y_a = x_gt_a + np.random.normal(0, std, len(x_gt_a))
        y_b = x_gt_b + np.random.normal(0, std, len(x_gt_b))

        x1 = x_a
        y1 = y_a
        x2 = np.concatenate([x_a, x_b])
        y2 = np.concatenate([y_a, y_b])

        comparator1 = ContinuousComparator(reference_method_measurements=x1,
                                           new_method_measurements=y1,
                                           weighting=weighting,
                                           reference_method_type="soft",
                                           soft_regression_method="BA")
        comparator2 = ContinuousComparator(reference_method_measurements=x2,
                                           new_method_measurements=y2,
                                           weighting=weighting,
                                           reference_method_type="soft",
                                           soft_regression_method="BA")

        fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
        comparator1.plot_regression(ax=axs[0], title="Balanced")
        comparator2.plot_regression(ax=axs[1], title="Unbalanced")

        # Allow a 1% relative tolerance
        assert (np.isclose(comparator1.metrics["linreg_pearson_r"],
                           comparator2.metrics["linreg_pearson_r"],
                           atol=0,
                           rtol=0.01))
        self.show_fig(fig, title=self._testMethodName)

    def test_out_of_range_of_interest_data(self):
        # This test should not display regression line as no points are in the range of interest
        mean = 50
        x_gt = np.random.normal(mean, 10, 1000)
        std_x = 1.
        std_y = 1.
        fixed_bias = 160
        proportional_bias = 0
        x = x_gt + np.random.normal(0, std_x, len(x_gt))
        y = x_gt * proportional_bias + fixed_bias + np.random.normal(0, std_y, len(x_gt))

        comparator = ContinuousComparator(reference_method_measurements=x,
                                          new_method_measurements=y,
                                          range_of_interest=[20, 80],
                                          binwidth=1,
                                          reference_method_type="soft",
                                          weighting="inverse")

        fig = comparator.plot_data_distribution(xlim=[0, 150], return_fig=True)
        self.show_fig(fig, title=self._testMethodName)
        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_regression(xlim=[0, 200],
                                   ylim=[0, 200],
                                   ax=ax)
        self.show_fig(fig, title=self._testMethodName)
        fig, ax = plt.subplots(figsize=(8, 8))
        comparator.plot_bland_altman(xlim=[0, 200],
                                     ylim=[-200, 200],
                                     ax=ax)
        self.show_fig(fig, title=self._testMethodName)

    @staticmethod
    def _show_fig(fig=None, timeout=2000, title=None):
        if timeout:
            timer = fig.canvas.new_timer(interval=timeout)
            timer.add_callback(plt.close)

        if title:
            if "test" in title:
                title = _format_test_method_name(title)
            fig.suptitle(title)

        if timeout:
            timer.start()

        plt.tight_layout()
        plt.show()


def _format_test_method_name(name):
    return name.replace("test_", "").replace("_", " ").capitalize()
