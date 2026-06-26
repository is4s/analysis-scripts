import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def get_statistics(arr: np.ndarray):
    """
    Generate the following statistics from a given array of values:
    1) Mean
    2) Standard Deviation
    3) Root Mean Square
    4) Absolute Max

    Args:
        - arr (np.ndarray): Array for which to generate statistics.

    Returns:
        - String containing the generated statistics.
    """
    m = np.nanmean(arr)
    std = np.nanstd(arr)
    rms = np.sqrt(np.nansum(np.square(arr)) / np.count_nonzero(np.isfinite(arr)))
    abs_max = np.nanmax(np.abs(arr))

    statistics_str = (
        f'Mean: {m:.2f}\n'
        + f'Standard Deviation: {std:.2f}\n'
        + f'Root Mean Square: {rms:.2f}\n'
        + f'Absolute Max: {abs_max:.2f}'
    )

    return statistics_str


def show_stats(fig: Figure, y: np.ndarray):
    plt.subplots_adjust(right=0.7)
    ax_pos = plt.gca().get_position()
    x_pos = ax_pos.xmax + 0.01
    y_pos = ax_pos.ymin + 0.01
    stats = get_statistics(y)
    fig.text(x_pos, y_pos, stats)
