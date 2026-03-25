from matplotlib.axes import Axes
from numpy.typing import NDArray


class PlotData:
    label: str
    x: NDArray
    y: list[NDArray]

    is_scatter: bool
    kwargs: dict

    def __init__(self, label, x, y, is_scatter=False, **kwargs):
        self.label = label
        self.x = x
        self.y = y
        self.is_scatter = is_scatter
        self.kwargs = kwargs

    def _scatter(self, ax: Axes, y_idx: int):
        ax.scatter(self.x, self.y[y_idx], label=self.label, **self.kwargs)

    def _line(self, ax: Axes, y_idx: int):
        ax.plot(self.x, self.y[y_idx], label=self.label, **self.kwargs)

    def plot(self, ax: Axes, y_idx: int = 0):
        if self.is_scatter:
            self._scatter(ax, y_idx)
        else:
            self._line(ax, y_idx)
