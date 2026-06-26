import numpy as np
from aspn23_xtensor import to_seconds
from navanalysis.lcm.conversions import llh_to_ned
from navanalysis.lcm.data import PosData

from .LogReader import LogReader


class PosLogReader(LogReader[PosData]):
    llh0: np.ndarray | None

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
            data.sig = np.array(data.sig)

            if self.llh0 is None:
                self.llh0 = data.llh[0]
            data.ned = llh_to_ned(data.llh, self.llh0)
