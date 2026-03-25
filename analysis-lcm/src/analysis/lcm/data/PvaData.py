import numpy as np

from analysis.lcm.conversions import llh_to_ned, ned_sigma_to_llh_sigma
from analysis.lcm.measurements import get_pva

from .Data import Data


class PvaData(Data):
    llh: list | np.ndarray
    vel: list | np.ndarray
    rpy: list | np.ndarray
    llh_sig: list | np.ndarray
    ned_sig: list | np.ndarray
    vel_sig: list | np.ndarray
    tilt_sig: list | np.ndarray

    def __init__(self, label):
        super().__init__(label)
        self.llh = []
        self.vel = []
        self.rpy = []
        self.llh_sig = []
        self.ned_sig = []
        self.vel_sig = []
        self.tilt_sig = []

    def add_data(self, time, aspn_msg):
        llh, vel, rpy, ned_sig, vel_sig, tilt_sig = get_pva(aspn_msg)

        if llh is not None and ned_sig is not None:
            self.time.append(time)
            self.llh.append(llh)
            self.vel.append(vel)
            self.ned_sig.append(ned_sig)
            self.vel_sig.append(vel_sig)
            if tilt_sig.size:
                self.rpy.append(rpy)
                self.tilt_sig.append(tilt_sig)

    def set_ned_pos(self, llh0=None):
        self.ned = llh_to_ned(self.llh, llh0)

    def set_llh_sigma(self):
        self.llh_sig = ned_sigma_to_llh_sigma(self.ned_sig, self.llh)
