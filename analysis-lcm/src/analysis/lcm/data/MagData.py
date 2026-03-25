import numpy as np

from analysis.lcm.data import Data
from analysis.lcm.measurements import get_heading, get_mag


class MagData(Data):
    mag: list | np.ndarray
    scale_factor: list | None
    bias: list | None
    heading: list | np.ndarray

    def __init__(self, label):
        super().__init__(label)
        self.mag = []
        self.heading = []
        self.scale_factor = None
        self.bias = None

    def add_data(self, time, aspn_msg):
        # Save mag or heading data
        mag, _ = get_mag(aspn_msg)
        heading, _ = get_heading(aspn_msg)

        self.time.append(time)
        if mag is not None:
            self.mag.append(mag)
        if heading is not None:
            self.heading.append(heading)
