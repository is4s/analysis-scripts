#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from navanalysis.lcm.data import ImuData, LogData
from navanalysis.lcm.interpolation import sum_imu
from navanalysis.lcm.log_readers import read_imu
from navanalysis.lcm.plots import Plot
from navtk.navutils import calculate_gravity_schwartz, rpy_to_dcm


def plot_stationary_imu(
    log_data: LogData[ImuData], alt: float, lat_deg: float, heading_deg: float
) -> None:
    t0 = log_data.t0
    imu_data = log_data.data

    # Calculate expected rotation rate
    omega = 7.2921151467e-5
    lat = np.deg2rad(lat_deg)
    earth_rate_ned = np.array([np.cos(lat), 0, -np.sin(lat)]) * omega
    rpy = np.deg2rad([0, 0, heading_deg])
    C_imu_to_ned = rpy_to_dcm(rpy)
    earth_rate_imu = C_imu_to_ned.T @ earth_rate_ned

    # Calculate expected gravity
    g = calculate_gravity_schwartz(alt, lat)

    # Plot IMU vs. time
    accel_plot = Plot(
        'DV',
        f'Time (s), t0 = {t0}',
        ['X (m/s)', 'Y (m/s)', 'Z (m/s)'],
    )
    gyro_plot = Plot(
        'DTH',
        f'Time (s), t0 = {t0}',
        ['X (rad)', 'Y (rad)', 'Z (rad)'],
    )
    alpha = 1
    for channel, data in imu_data.items():
        if channel == log_data.truth_channel:
            continue
        accel_plot.add_data(channel, data.time, data.accel, alpha=alpha)
        gyro_plot.add_data(channel, data.time, data.gyro, alpha=alpha)
        alpha = 0.3

    expected_time = accel_plot.data[-1].x
    N = len(expected_time)
    g = np.tile(g, (N, 1))
    earth_rate_imu = np.tile(earth_rate_imu, (N, 1))
    mean_dt = np.mean(np.diff(expected_time))

    accel_plot.add_data('Gravity', expected_time, -g * mean_dt)
    gyro_plot.add_data('Earth Rate', expected_time, earth_rate_imu * mean_dt)

    accel_plot.plot()
    gyro_plot.plot()

    # Plot IMU error vs time
    shared_accel_plot = Plot(
        'Accel',
        f'Time (s), t0 = {t0}',
        ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
    )
    shared_gyro_plot = Plot(
        'Gyro',
        f'Time (s), t0 = {t0}',
        ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'],
    )
    for channel, data in imu_data.items():
        if channel == log_data.truth_channel:
            continue
        # Calc accelerations from IMU meas
        time, accel, gyro = sum_imu(data.time, data.accel, data.gyro, 200)

        shared_accel_plot.add_data(channel, time, accel)
        shared_gyro_plot.add_data(channel, time, gyro)

        accel_error = accel + g[0]
        gyro_error = gyro - earth_rate_imu[0]
        accel_err_plot = Plot(
            title=f'Accel Bias for {channel}',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
        )
        accel_err_plot.add_data(channel, time, accel_error, is_scatter=True, marker='.')
        accel_err_plot.plot()

        gyro_err_plot = Plot(
            title=f'Gyro Bias for {channel}',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'],
        )
        gyro_err_plot.add_data(channel, time, gyro_error, is_scatter=True, marker='.')
        gyro_err_plot.plot()

    shared_accel_plot.add_data('Gravity', expected_time, -g)
    shared_gyro_plot.add_data('Earth Rate', expected_time, earth_rate_imu)

    shared_accel_plot.plot()
    shared_gyro_plot.plot()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="""Plot stationary IMU messages from LCM log and compare to expected measurements derived from earth rate."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-l',
        '--lat',
        help='Approximate latitude of IMU in degrees, used to compute earth rate at location.',
        type=float,
    )
    parser.add_argument(
        '-a',
        '--alt',
        help='Approximate altitude of IMU in meters, used to compute gravity at location. Defaults to 0.',
        default=0.0,
        type=float,
    )
    parser.add_argument(
        '-y',
        '--yaw',
        help='Yaw (heading) of IMU in degrees, relative to true North. Used to rotate earth rate into IMU frame.',
        type=float,
    )

    args = parser.parse_args()
    log_data = read_imu(args.logfile, False)
    plot_stationary_imu(log_data, args.alt, args.lat, args.yaw)


if __name__ == '__main__':
    main()
