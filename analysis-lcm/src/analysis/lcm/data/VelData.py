import numpy as np

from analysis.lcm.data import Data
from analysis.lcm.measurements import get_vel


class VelData(Data):
    vel: list | np.ndarray
    sig: list | np.ndarray

    def __init__(self, label):
        super().__init__(label)
        self.vel = []
        self.sig = []

    def add_data(self, time, aspn_msg):
        # Save vel message
        vel, sig = get_vel(aspn_msg)
        self.time.append(time)
        self.vel.append(vel)
        self.sig.append(sig)
