import numpy as np

from analysis.lcm.data import Data


class AltData:
    time: list | np.ndarray
    alt: list | np.ndarray
    sigma: list | np.ndarray
    bias: float
    temp_time: list | np.ndarray
    temp: list | np.ndarray
    ref_pressure: float
    ref_alt: float

    def __init__(self, label, bias=0):
        # super().__init__(label)
        self.time = []
        self.alt = []
        self.sigma = []
        self.temp_time = []
        self.temp = []
        self.bias = bias
        self.ref_pressure = 101325.0
        self.ref_alt = 0.0
