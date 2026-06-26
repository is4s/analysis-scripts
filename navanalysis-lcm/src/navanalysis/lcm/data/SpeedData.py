import numpy as np
from navanalysis.lcm.measurements import get_speed

from .Data import Data


class SpeedData(Data):
    speed: list | np.ndarray

    def __init__(self, label):
        super().__init__(label)
        self.speed = []

    def add_data(self, time, aspn_msg):
        speed, _ = get_speed(aspn_msg)

        # Save speed data
        self.time.append(time)
        self.speed.append(speed)
