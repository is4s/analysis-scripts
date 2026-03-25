import numpy as np

# from analysis.lcm import *
from analysis.lcm.measurements import get_imu

from .PvaData import Data


class ImuData(Data):
    accel: list | np.ndarray
    gyro: list | np.ndarray

    def __init__(self, label):
        super().__init__(label)
        self.accel = []
        self.gyro = []

    def add_data(self, time, aspn_msg):
        # Save data from IMU message
        accel, gyro = get_imu(aspn_msg)
        self.time.append(time)
        self.accel.append(accel)
        self.gyro.append(gyro)
