import numpy as np
from aspn23_xtensor import to_seconds

from analysis.lcm.data import RangeRateData, VelData

from .LogReader import LogReader


class RangeRateLogReader(LogReader[RangeRateData | VelData]):
    def __init__(
        self, logfile: str, desired_types: tuple, save_all: bool, config_file: str
    ):
        super().__init__(logfile, RangeRateData, desired_types, save_all, config_file)
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = VelData(
            self.log_data.truth_channel
        )

    def postprocess(self):
        for channel, data in self.log_data.data.items():
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])

            if channel == self.log_data.truth_channel:
                data.vel = np.array(data.vel)
            else:
                data.range_rates = [np.array(rates) for rates in data.range_rates]
                data.rcs = [np.array(rates) for rates in data.rcs]
                data.power = [np.array(rates) for rates in data.power]
                data.noise = [np.array(rates) for rates in data.noise]
