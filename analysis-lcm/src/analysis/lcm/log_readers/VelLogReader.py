import numpy as np
from aspn23_xtensor import to_seconds

from analysis.lcm.data import VelData

from .LogReader import LogReader


class VelLogReader(LogReader[VelData]):
    def __init__(
        self, logfile: str, desired_types: tuple, save_all: bool, config_file: str
    ):
        super().__init__(logfile, VelData, desired_types, save_all, config_file)
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = self.new_data(
            self.log_data.truth_channel
        )

    def postprocess(self):
        for channel, data in self.log_data.data.items():
            # Convert arrays to np, and make timestamps relative
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])
            data.vel = np.array(data.vel)
            data.sig = np.array(data.sig)

            # Rotate IMU measurements into platform frame
            C_sensor_to_platform = np.eye(3)
            if channel in self.config and 'sensor_to_platform' in self.config[channel]:
                C_sensor_to_platform = self.config[channel]['sensor_to_platform']
            data.vel = (C_sensor_to_platform @ data.vel.T).T
