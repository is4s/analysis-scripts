#!/usr/bin/env python3
import argparse

import numpy as np
from analysis.lcm.data import ImuData, SpeedData
from analysis.lcm.log_readers import ImuLogReader
from analysis.lcm.measurements import decode_aspn_lcm_msg, get_speed
from aspn23_lcm import measurement_IMU
from aspn23_xtensor import to_seconds
from lcm import Event, EventLog
from navtk import solve_wahba_svd


def get_imu_data(logfile, use_all):
    """
    Get IMU measurements from the LCM log and return as a dictionary

    :param logfile: Path to LCM log
    :type logfile: str
    :param use_all: Log all available IMUs to the dictionary. If `False`, CLI prompts will appear
    to ask which IMUs the user wants to log.
    :type use_all: bool
    :return: ImuData
    :rtype: dict
    """
    log_reader = ImuLogReader(logfile, (measurement_IMU,), use_all)
    imu_data: dict[str, ImuData] = log_reader.read_log()
    t0 = log_reader.log_data.t0

    return imu_data, t0


def get_speed_data(logfile):
    """
    Get platform's speed measurements (here hardcoded to INS-D measurements) from the LCM log and return as a dictionary

    :param logfile: Path to LCM log
    :type logfile: str
    :return: speed_data
    :rtype: SpeedData
    """
    speed_data = SpeedData('/sensor/xterra/speed')
    log = EventLog(logfile, 'r')

    msg: Event
    for msg in log:
        if speed_data.label == msg.channel:
            t, aspn_msg = decode_aspn_lcm_msg(msg)
            s, _ = get_speed(aspn_msg)
            speed_data.time.append(t)
            speed_data.speed.append(s * speed_data.scale)

    return speed_data


def get_imu_rotations(logfile, use_all):
    """
    Get IMU and platform speed measurements (typically from INS-D) from the input LCM log and iteratively print out
    the imu_to_platform rotation determined by NavToolkit's Wahba algorithm, solve_wahba_svd

    :param logfile: Path to LCM log
    :type logfile: str
    :param use_all: Log all available IMUs to the dictionary. If `False`, CLI prompts will appear
    to ask which IMUs the user wants to log.
    :type use_all: bool
    """
    imu_data, t0 = get_imu_data(logfile, use_all)
    truth_data = get_speed_data(logfile)

    for key, data in imu_data.items():
        time = np.array([to_seconds(t - t0) for t in data.time])
        truth_time = np.array([to_seconds(t - t0) for t in truth_data.time])

        # Grab truth speed at IMU times so we can calculate expected accelerations
        truth_data = get_speed_data(logfile)
        speed = np.interp(time, truth_time, truth_data.speed)

        avg_dt = np.mean(np.diff(time))

        # Calculate expected acceleration
        # Expected X = delta speed / delta time
        # Expected Y = 0
        # Expected Z = Gravity
        expected_accel = np.zeros((time.size - 1, 3))
        expected_accel[:, 0] = np.diff(speed / avg_dt)
        expected_accel[:, 2] = -(9.8 * avg_dt)

        # Get measured acceleration
        imu_accel = np.array(data.accel[1:])

        imu_to_platform = solve_wahba_svd(expected_accel, imu_accel)

        print(f'IMU to Platform rotation for {key}:\n{imu_to_platform}\n')


def main():
    parser = argparse.ArgumentParser(
        description="""Take IMU and INS-D speed messages from log and compute imu_to_platform rotation.
        Assumes the vehicle moves only in a straight line and has periods of static data before and after motion."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Compute imu-to-platform rotation for all IMU channels in log. If not set, will prompt user to determine which IMU channels should be used.',
        action='store_true',
    )

    args = parser.parse_args()
    get_imu_rotations(args.logfile, args.all)


if __name__ == '__main__':
    main()
