import numpy as np
from aspn23_xtensor import to_seconds

from analysis.lcm.data import SpeedData

from .LogReader import LogReader


class SpeedLogReader(LogReader[SpeedData]):
    def __init__(
        self, logfile: str, desired_types: tuple, save_all: bool, config_file: str
    ):
        super().__init__(logfile, SpeedData, desired_types, save_all, config_file)
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = self.new_data(
            self.log_data.truth_channel
        )

    def postprocess(self):
        for data in self.log_data.data.values():
            # Convert to np array
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])
            data.speed = np.array(data.speed)
