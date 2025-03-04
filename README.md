# UncertMedCompare: Comparison of Uncertain Clinical Measurement methods

A python package to simplify the comparison of clinical measurement methods, with focus on measurement uncertainty. 
Calculates several metrics and generate plots to easily quantify and visualize clinical value of new measurement 
methods.

## Why use UncertMedCompare for comparing measurement methods?

The idea behind UncertMedCompare is to provide:

- **Support for uncertain measurement methods**. Measured values may be estimates of the true value and have some 
uncertainty (referred as *soft*, for example heart Ejection Fraction). They may also be directly the true value 
(referred as *hard*, for example patient age). *Soft* measurement values are treated with Bland-Altman 
analyses that account for errors in the measurement methods. *Hard* measurement values are treated with Ordinary Least 
Squares regression. See 
[this article from Bland and Altman (section "Regression Lines")](https://doi.org/10.1002/uog.122) 
that explain why ordinary least squares regression lines should not be used when the reference values are uncertain.

- **Metrics that are independent of the test data**. This is achieved by using inverse weighting of the samples. This is 
illustrated in `examples/regression_metrics.py` where MAE, R2, etc. get independent of the data distribution when using 
inverse weighting.

- **Metrics that are calculated over a specific range of interest**. For a clinical measurement, it might be relevant to
compare methods within a specific range of values, for example corresponding to a specific group of patients or a 
specific clinical problem. UncertMedCompare allows to calculate metrics within a specified range of interest. This is 
illustrated in `examples/range_of_interest.py`

- **Isolate accuracy and precision metrics**. For measurement methods based on AI algorithms, it is important to isolate 
accuracy and precision. Indeed, the cost function when training an AI algorithm optimizes one scalar (which is often a 
proxy for the clinical task) although one seek for AI algorithms that are optimized for both accuracy and precision. In 
practice, both often gets optimized at the same time through backpropagation, but it is good for reporting and 
troubleshooting to quantify both.
    - Soft measurement values through Bland-Altman analysis (see `examples/bland_altman_analysis.py`):
        * Proportional bias (accuracy): Bland-Altman slope
        * Fixed bias (accuracy): Bland-Altman mean error
        * Random error (precision): Bland-Altman LoAs
    - Soft measurement values through Least Products regression (see `examples/least_product_regression.py`) or Bland-Altman regression (see `examples/bland_altman_regression.py`):
        * Proportional bias (accuracy): Regression slope
        * Fixed bias (accuracy): Regression intercept
        * Random error (precision): Pearson correlation coefficient
    - Hard measurement values through Least Squares regression (see `examples/least_square_regression.py`):
        * Proportional bias (accuracy): Regression slope
        * Fixed bias (accuracy): Regression intercept
        * Random error (precision): Pearson correlation coefficient. 

- **A model calibration functionality**. There can be a proportional and/or fixed bias between the tow measurement 
methods being compared. Assuming that the reference method is unbiased, it is possible to calibrate the new method to 
remove its bias. This is illustrated in `examples/calibration.py`, for both hard reference values (the calibration is 
performed from a regression plot point of view) and for soft reference values (the calibration is performed from a 
Aland-Altman point of view).

NOTE: The pearson correlation coefficient is very dependent on the range of the values, which is an issue when comparing 
across different datasets. This limitation of the pearson correlation coefficient is explained in [this other article 
from Bland and Altman (section "The interpretation of correlation coefficients")](https://doi.org/10.1002/uog.122).
Specifying a `range_of_interest` and setting `weighting="inverse"` can mitigate this issue.

## How does UncertMedCompare works?

```python
import numpy as np
import matplotlib.pyplot as plt
from UncertMedCompare.continuous_comparator import ContinuousComparator

gt = np.random.uniform(20, 80, 1000)
x = gt + np.random.normal(0, 2, len(gt))
y = gt + np.random.normal(0, 2, len(gt))
comparator = ContinuousComparator(reference_method_measurements=x,  # Measurements with the reference method
                                  new_method_measurements=y,  # Measurements with the new method
                                  range_of_interest=[30, 70], # Range of values to calculate the metrics on. Default: None
                                  binwidth=1,  # Discretize the values
                                  reference_method_type="soft",  # "soft" or "hard"
                                  weighting="inverse",  # Weighting strategy. Default is None
                                  limit_of_agreement=1.96,  # LoA for the Bland-Altman plot
                                  confidence_interval=95, # Confidence interval for all metrics. Default 95% two sided
                                  bootstrap_samples=1000)  # Number of bootstrap samples. Default is 10000.

comparator.fit_linear_regression()  # Calculate regression slope and intercept
comparator.fit_bland_altman()  # Calculate mean error and LoAs with CIs
comparator.calculate_error_metrics()  # Calculate mean error and std
comparator.calculate_regression_metrics()  # Calculate MAE, R2, RMSE
comparator.bootstrap_error_metrics()  # Calculate mean error and std with CIs
comparator.bootstrap_regression_metrics()  # Calculate MAE, R2, RMSE with CIs

print(comparator.metrics)  # Get all the calculated metrics
print(comparator.heteroscedasticity_info)  # Get heteroscedasticity info

# Plot regression and Bland-Altman analysis
fig, axs = plt.subplots(ncols=2, figsize=(15, 7.5))
comparator.plot_regression(xlim=[0, 100], ylim=[0, 100],  # Plotlims
                           ax=axs[0],  # Axes object from matplotlib.pyplot
                           title="Regression example")  # Subplot title
comparator.plot_bland_altman(xlim=[0, 100], ylim=[-50, 50],
                             ax=axs[1],
                             title="Bland-Altman example")
fig.show()
plt.close(fig)

# Plot for qualitative check of heteroscedasticity
fig, ax = plt.subplots(figsize=(7.5, 7.5))
comparator.plot_heteroscedasticity(xlim=[0, 100], ylim=[0, 20],
                                   ax=ax,
                                   title="Heteroscedasticity investigation")
fig.show()
plt.close(fig)
```

Visit the `examples` for more  details.

## Good to know

**NOTE 1:** Inverse weighting is the extension of [balanced accuracy](https://doi.org/10.1109/ICPR.2010.764) from classification tasks to regression 
task. That simple.

**NOTE 2:** When the magnitude of the random error in the reference and new values is different, there appears a 
linear trend in the Bland-Altman plot [as explained here](https://doi.org/10.1016/j.gloepi.2020.100045).

**NOTE 3:** UncertMedCompare has an implementation of [Ordinary Least Products](https://doi.org/10.1111/j.1440-1681.2010.05376.x) (also called Reduced Major Axis) 
regression that accounts for soft reference values. Alternatively, regression can be performed in the Bland-Altman space 
(difference vs. means) to account for the soft reference values. Geometrically, the Bland-Altman plot is a simple 45 
degrees rotation of the Method A vs. Method B plot.

**NOTE 4:** All major functionalities come together with a test. Running the tests is also a good way to understand how 
the package works.

## Installation 

If you want to use the package in other projects, this can be achieved by installing the
package directly from the git repository or from source. Use this steps if you plan 
to use the package without developing, running examples or tests.

**Option 1: Directly from repo using pip install**

First ensure that pip is up-to-date:
```bash
pip install --upgrade pip # Make sure pip is up to date first
```

Install the package directly from the bitbucket repository
```bash
pip install git+https://bitbucket.org/ntnuultrasoundgroup/uncertmedcompare.git
```

**Option 2: From source**

You can also install it from source by downloading the repo from either github or bitbucket
```bash
git clone https://bitbucket.org/ntnuultrasoundgroup/uncertmedcompare.git
cd UncertMedCompare
pip install -e .
```

## Setup for development
Follow this step if you want to run examples, tests or contribute to the code.

**1. Clone repo**
```bash
git clone https://bitbucket.org/ntnuultrasoundgroup/uncertmedcompare.git
```

**2. Setup virtual environment**
Enter the UncertMedCompare source code

```bash
cd UncertMedCompare
```

Run the following to setup a virtual environment:

*Ubuntu:*
```bash
virtualenv --python=python3 venv
source venv/bin/activate
```

Windows:
```bash
python3 -m virtualenv venv
.\venv\Scripts\Activate.ps1
```

A guide for packages with pip and virtual environments can be found 
[here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

**3. Install requirements**
```bash
pip install --upgrade pip # Make sure pip is up to date first
pip install -r requirements.txt
pip install -e .
```

## Examples and tests

Once UncertMedCompare is setup for development, you can try to run various examples and tests.

**Run example:**
```bash
cd examples
python3 least_square_regression.py
```

**Run tests:**
```bash
cd tests
python3 run_all_tests.py
```

## Contact

[David Pasdeloup](https://www.linkedin.com/in/david-pasdeloup-a2166712b/) (david.pasdeloup@ntnu.no)