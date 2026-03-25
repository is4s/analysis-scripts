import aspn23_lcm
import numpy as np

from analysis.lcm.data import Data


def get_range_rate(aspn_msg):
    range_rate = None
    sig = None
    if isinstance(aspn_msg, aspn23_lcm.measurement_range_rate_to_point):
        range_rate = aspn_msg.obs
        sig = np.sqrt(aspn_msg.variance)

    return range_rate, sig


def get_point_pos(aspn_msg):
    if isinstance(aspn_msg, aspn23_lcm.measurement_range_rate_to_point):
        return [
            aspn_msg.remote_point.position1,
            aspn_msg.remote_point.position2,
            aspn_msg.remote_point.position3,
        ]


def get_id(aspn_msg):
    if isinstance(aspn_msg, aspn23_lcm.measurement_range_rate_to_point):
        return aspn_msg.remote_point.id


class RangeRateData(Data):
    # 2D lists, where inner list is all the points for a given time step
    range_rates: list[list] | np.ndarray
    ids: list[list] | np.ndarray
    points: list[list[list]] | np.ndarray
    rcs: list[list]
    power: list[list]
    noise: list[list]

    def __init__(self, label):
        super().__init__(label)
        self.range_rates = []
        self.ids = []
        self.points = []
        self.rcs = []
        self.power = []
        self.noise = []

    def add_data(self, time, aspn_msg):
        range_rate, _ = get_range_rate(aspn_msg)
        id = get_id(aspn_msg)
        point_pos = get_point_pos(aspn_msg)
        rcs, power, noise = aspn_msg.error_model_params

        if len(self.time) == 0 or id <= self.ids[-1][-1]:
            # New time step
            self.time.append(time)
            self.range_rates.append([])
            self.ids.append([])
            self.points.append([])
            self.rcs.append([])
            self.power.append([])
            self.noise.append([])

        # New point for existing time step
        self.range_rates[-1].append(range_rate)
        self.ids[-1].append(id)
        self.points[-1].append(point_pos)
        self.rcs[-1].append(rcs)
        self.power[-1].append(power)
        self.noise[-1].append(noise)
