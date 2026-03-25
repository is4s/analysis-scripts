import numpy as np
from aspn23_xtensor import to_seconds

from analysis.lcm.data import MagData

from .LogReader import LogReader


class MagLogReader(LogReader[MagData]):
    truth_channel: str

    def __init__(
        self, logfile: str, desired_types: tuple, save_all: bool, config_file: str
    ):
        super().__init__(logfile, MagData, desired_types, save_all, config_file)
        # Save truth messages
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = self.new_data(
            self.log_data.truth_channel
        )

    def postprocess(self):
        for data in self.log_data.data.values():
            # Convert arrays to np, and make timestamps relative
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])
            data.mag = np.array(data.mag)
            data.heading = np.array(data.heading)
