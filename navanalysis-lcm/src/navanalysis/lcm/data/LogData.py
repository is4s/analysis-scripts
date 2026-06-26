from typing import Generic

from aspn23_xtensor import TypeTimestamp

from .Data import DataType


class LogData(Generic[DataType]):
    logfile: str
    truth_channel: str
    t0: TypeTimestamp
    data: dict[str, DataType]

    def __init__(self, logfile: str):
        self.logfile = logfile
        self.data = {}
        self.t0 = None
        self.truth_channel = None
