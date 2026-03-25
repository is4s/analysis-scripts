import inu
import numpy as np
from aspn23_xtensor import to_seconds

from analysis.lcm.data import ImuData, PvaData
from analysis.lcm.interpolation import interpolate_pva_advanced

from .LogReader import LogReader


class ImuLogReader(LogReader[ImuData | PvaData]):
    TRUTH_IMU_DT = 0.01

    def __init__(
        self,
        logfile: str,
        desired_types: tuple,
        save_all: bool,
        config_file: str,
    ):
        super().__init__(logfile, ImuData, desired_types, save_all, config_file)
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = PvaData(
            self.log_data.truth_channel
        )

    def postprocess(self):
        """Convert truth PvaData to ImuData via inverse mechanization."""

        # Convert arrays to np, and make timestamps relative
        for channel, data in self.log_data.data.items():
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])

            if channel == self.log_data.truth_channel:
                continue

            data.accel = np.array(data.accel)
            data.gyro = np.array(data.gyro)

            # Rotate IMU measurements into platform frame
            C_imu_to_platform = np.eye(3)
            if channel in self.config and 'sensor_to_platform' in self.config[channel]:
                C_imu_to_platform = self.config[channel]['sensor_to_platform']
            data.accel = (C_imu_to_platform @ data.accel.T).T
            data.gyro = (C_imu_to_platform @ data.gyro.T).T

        # Downsample PVAs to 5 Hz. This is necessary to ensure interpolation to a higher
        # frequency is smooth.
        pva_data: PvaData = self.log_data.data[self.log_data.truth_channel]
        desired_dt = 0.2
        avg_dt = np.mean(np.diff(pva_data.time))
        if avg_dt < desired_dt:
            downsample_factor = int(desired_dt / avg_dt)
            pva_data.time = pva_data.time[::downsample_factor]
            pva_data.llh = pva_data.llh[::downsample_factor]
            pva_data.vel = pva_data.vel[::downsample_factor]
            pva_data.rpy = pva_data.rpy[::downsample_factor]

        # Evenly space timestamps
        pva_data.time = np.linspace(
            pva_data.time[0], pva_data.time[-1] + desired_dt, len(pva_data.time)
        )

        # Interpolate PVA to 100 Hz
        interp_pva, interp_time = interpolate_pva_advanced(
            llh_t=np.hstack((pva_data.time[:, np.newaxis], pva_data.llh)),
            rpy_t=np.hstack((pva_data.time[:, np.newaxis], pva_data.rpy)),
            dt=0.01,
            vel_t=np.hstack((pva_data.time[:, np.newaxis], pva_data.vel)),
            interp_type='spline',
        )
        # Calculate "true" sampled IMU measurements via inverse mechanization
        forces, rates = inu.inv_mech(interp_pva[:, :3], interp_pva[:, 6:9], 0.01)
        self.log_data.data[self.log_data.truth_channel] = self.new_data(
            self.log_data.truth_channel
        )
        self.log_data.data[self.log_data.truth_channel].time = interp_time
        self.log_data.data[self.log_data.truth_channel].accel = forces * 0.01
        self.log_data.data[self.log_data.truth_channel].gyro = rates * 0.01
