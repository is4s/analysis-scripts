import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .PlotData import PlotData


class Plot:
    title: str
    legend: str | None
    xlabel: str
    ylabels: list[str]

    data: list[PlotData]

    # Plot parameters
    equal: bool

    def __init__(self, title, xlabel, ylabels, legend=None, equal=False):
        self.title = title
        self.xlabel = xlabel
        if isinstance(ylabels, str):
            self.ylabels = [ylabels]
        else:
            self.ylabels = ylabels
        self.legend = legend
        self.equal = equal
        self.data = []

    def add_data(
        self, label: str, x: NDArray, y: list[NDArray], is_scatter=False, **kwargs
    ):
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = [y]
            elif y.ndim == 2 and y.shape[1] != x.size:
                # Ensure y is N x M, where N=number of subplots and M=number of points
                y = y.T

        self.data.append(PlotData(label, x, y, is_scatter=is_scatter, **kwargs))

    def plot(self, save_dir: str | None = None):
        num_subplots = len(self.ylabels)
        if num_subplots == 1:
            # One plot
            plt.figure(self.title)
            plt.suptitle(self.title)
            for data in self.data:
                data.plot(plt.gca())
            plt.ylabel(self.ylabels[0])
        else:
            axes: list[Axes]
            fig, axes = plt.subplots(
                num=self.title, nrows=num_subplots, ncols=1, sharex=True
            )
            # Plot data for one subplot at a time
            for y_idx in range(num_subplots):
                for data in self.data:
                    data.plot(axes[y_idx], y_idx)
                axes[y_idx].set_ylabel(self.ylabels[y_idx])

        plt.suptitle(self.title)
        plt.xlabel(self.xlabel)
        if self.equal:
            plt.axis('equal')
        if self.legend:
            plt.legend(self.legend)
        else:
            plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, self.title)
            plt.savefig(f'{filename}.png', dpi=300)
