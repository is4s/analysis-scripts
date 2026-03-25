import numpy as np
from aspn23_xtensor import to_seconds
from numpy import float64
from numpy.typing import NDArray

from analysis.lcm.data import PvaData

from .LogReader import LogReader


class PvaLogReader(LogReader[PvaData]):
    # For converting LLH for all PVAs to NED relative to the same initial pos
    llh0: NDArray[float64] | None

    def __init__(
        self,
        logfile: str,
        desired_types: tuple,
        save_all: bool,
        truth_channel: str | None = None,
        config_file: str = '',
    ):
        super().__init__(logfile, PvaData, desired_types, save_all, config_file)
        if truth_channel is not None:
            self.log_data.truth_channel = truth_channel
        else:
            self.log_data.truth_channel = self.config.get('truth_pva_channel', None)

        if truth_channel is not None:
            self.log_data.data[self.log_data.truth_channel] = self.new_data(
                self.log_data.truth_channel
            )
        self.llh0 = None

    def postprocess(self):
        if self.log_data.truth_channel is not None:
            truth_data = self.log_data.data[self.log_data.truth_channel]
            if len(truth_data.llh) > 0:
                self.llh0 = truth_data.llh[0]

        for data in self.log_data.data.values():
            # Convert tov to relative time
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])

            data.llh = np.array(data.llh)
            data.set_ned_pos(llh0=self.llh0)
            data.ned_sig = np.array(data.ned_sig)
            data.set_llh_sigma()

            data.vel = np.array(data.vel)
            data.vel_sig = np.array(data.vel_sig)

            data.rpy = np.rad2deg(np.unwrap(data.rpy, axis=0))
            data.tilt_sig = np.rad2deg(data.tilt_sig)
