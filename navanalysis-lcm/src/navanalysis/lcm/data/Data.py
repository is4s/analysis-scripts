from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from aspn23 import TypeTimestamp


class Data(ABC):
    label: str
    time: list | np.ndarray

    def __init__(self, label):
        self.label = label
        self.time = []

    @abstractmethod
    def add_data(self, time: TypeTimestamp, aspn_msg):
        pass


DataType = TypeVar('DataType', bound=Data)
