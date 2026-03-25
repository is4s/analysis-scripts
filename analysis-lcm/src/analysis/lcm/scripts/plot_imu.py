#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import ImuData, LogData
from analysis.lcm.interpolation import downsample_imu
from analysis.lcm.log_readers import read_imu
from analysis.lcm.plots import Plot
from scipy.interpolate import interp1d


def plot_imu(log_data: LogData[ImuData]) -> None:
    t0 = log_data.t0
    truth_channel = log_data.truth_channel
    imu_data = log_data.data

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
        accel_plot.add_data(channel, data.time, data.accel, alpha=alpha)
        gyro_plot.add_data(channel, data.time, data.gyro, alpha=alpha)
        alpha = 0.3
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
    shared_accel_err_plot = Plot(
        title='Accel Error',
        xlabel=f'Time (s) (t0 = {t0})',
        ylabels=['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
    )
    shared_gyro_err_plot = Plot(
        title='Gyro Error',
        xlabel=f'Time (s) (t0 = {t0})',
        ylabels=['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'],
    )
    for channel, data in imu_data.items():
        if channel == truth_channel:
            continue

        # Calc IMU error
        truth = imu_data[truth_channel]
        time, accel, gyro = downsample_imu(data.time, data.accel, data.gyro, 1.0)

        interp_accel_func = interp1d(
            truth.time,
            truth.accel / 0.01,
            axis=0,
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_gyro_func = interp1d(
            truth.time,
            truth.gyro / 0.01,
            axis=0,
            bounds_error=False,
            fill_value=np.nan,
        )
        truth_accel = interp_accel_func(time)
        truth_gyro = interp_gyro_func(time)
        accel_error = accel - truth_accel
        gyro_error = gyro - truth_gyro

        shared_accel_plot.add_data(channel, time, accel)
        shared_gyro_plot.add_data(channel, time, gyro)
        shared_accel_plot.add_data(truth.label, time, truth_accel)
        shared_gyro_plot.add_data(truth.label, time, truth_gyro)

        # IMU errors all on one plot
        shared_accel_err_plot.add_data(channel, time, accel_error)
        shared_gyro_err_plot.add_data(channel, time, gyro_error)

        # Individual IMU error plot for each channel
        accel_err_plot = Plot(
            title=f'Accel Error for {channel}',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'],
        )
        accel_err_plot.add_data(channel, time, accel_error, is_scatter=True, marker='.')
        accel_err_plot.plot()

        gyro_err_plot = Plot(
            title=f'Gyro Error for {channel}',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'],
        )
        gyro_err_plot.add_data(channel, time, gyro_error, is_scatter=True, marker='.')
        gyro_err_plot.plot()

    shared_accel_plot.plot()
    shared_gyro_plot.plot()
    shared_accel_err_plot.plot()
    shared_gyro_err_plot.plot()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="""Plot IMU messages from LCM log.""")

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all IMU messages in log. If not set, will prompt user to determine which IMU channels should be plotted.',
        action='store_true',
    )

    args = parser.parse_args()
    log_data = read_imu(args.logfile, args.all)
    plot_imu(log_data)


if __name__ == '__main__':
    main()
