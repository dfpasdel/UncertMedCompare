import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from unittest import TestCase


class TestUtilities(TestCase):
    def setUp(self):
        self.timeout_plots = 1500  # Set None/False if you want to avoid closing plots

        self.show_fig = partial(self._show_fig, timeout=self.timeout_plots)

    def tearDown(self):
        plt.close()

    def test_rolling_mean(self):
        from UncertMedCompare.utilities.misc import rolling_mean
        x = np.arange(0, 20, 1)
        y = np.zeros(len(x))
        y[5:15] = 1.
        fig = plt.figure()
        plt.plot(x, y, label="original")
        plt.plot(x, rolling_mean(y, n=5), label="rolling mean")
        plt.legend()
        self.show_fig(fig, title=self._testMethodName)

    @staticmethod
    def _show_fig(fig, timeout=2000, title=None):
        fig.set_size_inches(7.5, 5)
        if timeout:
            timer = fig.canvas.new_timer(interval=timeout)
            timer.add_callback(plt.close)

        if title:
            if "test" in title:
                title = _format_test_method_name(title)
            fig.suptitle(title)

        if timeout:
            timer.start()

        plt.show()


def _format_test_method_name(name):
    return name.replace("test_", "").replace("_", " ").capitalize()
