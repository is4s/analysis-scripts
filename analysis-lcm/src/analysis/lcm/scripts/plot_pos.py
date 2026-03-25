#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import LogData, PosData
from analysis.lcm.interpolation import interpolate_array
from analysis.lcm.log_readers import read_pos
from analysis.lcm.plots import Plot
from scipy.interpolate import interp1d


def pressure_to_alt(pressure, deg_k=288.15):
    alt = -(deg_k / 0.0065) * (
        pow(pressure / 101325.0, 8314.32 * 0.0065 / (9.80665 * 28.9644)) - 1.0
    )
    return alt


def plot_pos(log_data: LogData[PosData]) -> None:
    pos_data = log_data.data
    truth_channel = log_data.truth_channel
    t0 = log_data.t0

    # Plot horizontal trajectory
    traj_plot = Plot('Trajectory', 'Easting (m)', 'Northing (m)', equal=True)
    for channel, data in pos_data.items():
        traj_plot.add_data(
            channel,
            data.ned[:, 1],
            data.ned[:, 0],
            is_scatter=True,
            marker='.',
            # cmap='viridis',
            # c=data.time,
        )

        # Calculate horizontal distance traveled from truth position
        if channel == truth_channel:
            delta_pos_steps = np.linalg.norm(np.diff(data.ned[:, :2], axis=0), axis=1)
            delta_pos = np.sum(delta_pos_steps)
            print(f'Distance Traveled: {delta_pos:,.2f} m')
            delta_alt = np.sum(np.abs(np.diff(data.ned[:, 2])))
            print(f'Delta Alt: {delta_alt:.2f} m')
            avg_slope = delta_alt / delta_pos
            print(f'Average Slope: {avg_slope:.3f}')

    traj_plot.plot()

    # Plot position vs time
    pos_plot = Plot(
        'Position',
        f'Time (s), t0 = {t0}',
        ['Latitude (rad)', 'Longitude (rad)', 'Altitude (m)'],
    )
    for channel, data in pos_data.items():
        pos_plot.add_data(channel, data.time, data.llh, is_scatter=True, marker='.')
    pos_plot.plot()

    if len(pos_data.keys()) > 1 and len(pos_data[truth_channel].time) > 0:
        # Plot position error vs time
        shared_err = Plot(
            'Position Error',
            f'Time (s), t0 = {t0}',
            ['North (m)', 'East (m)', 'Down (m)'],
        )
        for channel, data in pos_data.items():
            if channel == truth_channel:
                continue

            # Calc position error
            truth = pos_data[truth_channel]
            truth_ned = interpolate_array(truth.time, truth.ned, data.time)
            error = data.ned - truth_ned

            # Individual position error w/ sigma plot for each channel
            shared_err.add_data(channel, data.time, error, is_scatter=True, marker='.')

            err_plot = Plot(
                f'Position Error for {channel}',
                f'Time (s), t0 = {t0}',
                ['North (m)', 'East (m)', 'Down (m)'],
                legend=[channel, '+/- 1 sigma'],
            )

            err_plot.add_data(channel, data.time, error, is_scatter=True, marker='.')
            err_plot.add_data(channel, data.time, data.sig, color='k', linestyle='--')
            err_plot.add_data(channel, data.time, -data.sig, color='k', linestyle='--')

            err_plot.plot()

        shared_err.plot()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="""Plot position messages from LCM log."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all position messages in log. If not set, will prompt user to determine which position channels should be plotted.',
        action='store_true',
    )

    args = parser.parse_args()
    log_data = read_pos(args.logfile, args.all)
    plot_pos(log_data)


if __name__ == '__main__':
    main()
