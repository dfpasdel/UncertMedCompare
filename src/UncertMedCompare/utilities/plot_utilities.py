import numpy as np

def boldify_legend_text(text):
    return "$\\bf{" + text.replace(" ", "\ ") + "}$"


def get_regression_line_soft_range_of_interest_intersections(linreg_slope, linreg_intercept, range_of_interest):
    center = 0.5 * (range_of_interest[0] + range_of_interest[1])
    half_diag = max(range_of_interest) - min(range_of_interest)
    x1 = (2 * center - half_diag - linreg_intercept) / (1 + linreg_slope)
    #x2 = (half_diag + linreg_intercept) / (1 - linreg_slope)
    #x3 = (half_diag - linreg_intercept) / (linreg_slope - 1)
    x4 = (2 * center + half_diag - linreg_intercept) / (1 + linreg_slope)
    x_crossing = []
    y_crossing = []
    """for x in [x1, x2, x3, x4]:
        if (x > center - half_diag) and (x < center + half_diag):
            y = linreg_slope * x + linreg_intercept
            if (y > center - half_diag) and (y < center + half_diag):
                x_crossing.append(x)
                y_crossing.append(y)
    return x_crossing, y_crossing"""
    for x in [x1, x4]:
        y = linreg_slope * x + linreg_intercept
        x_crossing.append(x)
        y_crossing.append(y)
    return x_crossing, y_crossing



