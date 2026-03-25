import numpy as np
from aspn23_xtensor import to_seconds

from analysis.lcm.conversions import llh_to_ned
from analysis.lcm.data import PosData

from .LogReader import LogReader


class PosLogReader(LogReader[PosData]):
    def __init__(
        self,
        logfile: str,
        desired_types: tuple,
        save_all: bool,
        config_file: str,
    ):
        super().__init__(logfile, PosData, desired_types, save_all, config_file)
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = self.new_data(
            self.log_data.truth_channel
        )

    def postprocess(self):
        for data in self.log_data.data.values():
            # Convert tov to relative time
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])
            data.llh = np.array(data.llh)
            data.sig = np.array(data.sig)
            data.ned = llh_to_ned(data.llh)
