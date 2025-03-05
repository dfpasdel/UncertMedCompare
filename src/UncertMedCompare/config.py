import matplotlib.pyplot as plt

# Plot style:
DEFAULT_STYLE = 'seaborn-v0_8-whitegrid'
if DEFAULT_STYLE not in plt.style.available:
    DEFAULT_STYLE = 'seaborn-whitegrid'
