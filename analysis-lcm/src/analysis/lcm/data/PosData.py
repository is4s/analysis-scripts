import numpy as np

from analysis.lcm.data import Data
from analysis.lcm.measurements import get_pos


class PosData(Data):
    llh: list | np.ndarray
    ned: np.ndarray
    sig: list | np.ndarray

    def __init__(self, label):
        super().__init__(label)
        self.llh = []
        self.sig = []
        self.ned = None

    def add_data(self, time, aspn_msg):
        pos, sig = get_pos(aspn_msg)

        if pos is not None and sig is not None:
            self.time.append(time)
            self.llh.append(pos)
            self.sig.append(sig)
