import os
from typing import Any, TypeVar

import aspn23_lcm

from analysis.lcm.config import CONFIG_FILE
from analysis.lcm.data import *
from analysis.lcm.log_readers import *

data_type_to_aspn_type: dict[type[DataType], tuple[Any, ...]] = {
    AltData: (
        aspn23_lcm.measurement_barometer,
        aspn23_lcm.measurement_altitude,
        aspn23_lcm.measurement_position,
        aspn23_lcm.measurement_position_velocity_attitude,
    ),
    ImuData: (
        aspn23_lcm.measurement_IMU,
    ),
    PosData: (
        aspn23_lcm.measurement_position,
        aspn23_lcm.measurement_position_velocity_attitude,
    ),
    RangeRateData: (aspn23_lcm.measurement_range_rate_to_point),
    PvaData: (
        aspn23_lcm.measurement_position_velocity_attitude,
    ),
    SpeedData: (
        aspn23_lcm.measurement_speed,
        aspn23_lcm.measurement_velocity,
        aspn23_lcm.measurement_delta_position,
    ),
    VelData: (
        aspn23_lcm.measurement_velocity,
        aspn23_lcm.measurement_position_velocity_attitude,
    ),
}


def read_alt(logfile: str, read_all: bool) -> LogData[AltData]:
    aspn_types = data_type_to_aspn_type[AltData]
    log_reader = AltLogReader(
        logfile,
        aspn_types,
        read_all,
        CONFIG_FILE,
    )
    return log_reader.read_log()


def read_imu(logfile: str, read_all: bool) -> LogData[ImuData]:
    aspn_types = data_type_to_aspn_type[ImuData]
    log_reader = ImuLogReader(logfile, aspn_types, read_all, CONFIG_FILE)
    return log_reader.read_log()


def read_pos(logfile: str, read_all: bool) -> LogData[PosData]:
    aspn_types = data_type_to_aspn_type[PosData]
    log_reader = PosLogReader(
        logfile,
        aspn_types,
        read_all,
        CONFIG_FILE,
    )
    return log_reader.read_log()


def read_range_rate_to_point(
    logfile: str, read_all: bool
) -> LogData[RangeRateData | VelData]:
    aspn_types = data_type_to_aspn_type[RangeRateData]
    log_reader = RangeRateLogReader(
        logfile,
        aspn_types,
        read_all,
        CONFIG_FILE,
    )
    return log_reader.read_log()


def read_pva(
    logfile: str, read_all: bool, truth_channel: str | None = None
) -> LogData[PvaData]:
    aspn_types = data_type_to_aspn_type[PvaData]
    log_reader = PvaLogReader(
        logfile,
        aspn_types,
        read_all,
        truth_channel=truth_channel,
        config_file=CONFIG_FILE,
    )
    return log_reader.read_log()


def read_speed(logfile: str, read_all: bool) -> LogData[SpeedData]:
    aspn_types = data_type_to_aspn_type[SpeedData]
    log_reader = SpeedLogReader(
        logfile,
        aspn_types,
        read_all,
        CONFIG_FILE,
    )
    return log_reader.read_log()


def read_vel(logfile: str, read_all: bool) -> LogData[VelData]:
    aspn_types = data_type_to_aspn_type[VelData]
    log_reader = VelLogReader(
        logfile,
        aspn_types,
        read_all,
        CONFIG_FILE,
    )
    return log_reader.read_log()
