#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import LogData, PvaData
from analysis.lcm.error import calc_tilts
from analysis.lcm.interpolation import interpolate_pva
from analysis.lcm.log_readers import read_pva
from analysis.lcm.plots import Plot


def plot_pva(log_data: LogData[PvaData], save=False) -> None:
    pva_data = log_data.data
    truth_channel = log_data.truth_channel
    t0 = log_data.t0

    save_dir = None
    if save:
        log_dir = os.path.dirname(log_data.logfile)
        log_name = os.path.basename(log_data.logfile).split('.')[0]
        save_dir = os.path.join(log_dir, log_name)

    # Plot horizontal trajectory
    traj_plot = Plot(
        title='Trajectory', xlabel='Easting (m)', ylabels='Northing (m)', equal=True
    )
    for channel, data in pva_data.items():
        traj_plot.add_data(
            data.label, data.ned[:, 1], data.ned[:, 0], is_scatter=True, marker='.'
        )

        # Calculate horizontal distance traveled from truth position
        if channel == truth_channel:
            delta_pos_steps = np.linalg.norm(np.diff(data.ned[:, :2], axis=0), axis=1)
            delta_pos = np.sum(delta_pos_steps)
            print(f'Distance Traveled: {delta_pos / 1000:.3f} km')
    traj_plot.plot(save_dir)

    # Plot position vs time
    pos_plot = Plot(
        title='Position',
        xlabel=f'Time (s) (t0 = {t0})',
        ylabels=['Latitude (rad)', 'Longitude (rad)', 'Altitude (m)'],
    )
    # Plot velocity vs time
    vel_plot = Plot(
        title='Velocity',
        xlabel=f'Time (s) (t0 = {t0})',
        ylabels=['North (m/s)', 'East (m/s)', 'Down (m/s)'],
    )
    # Plot attitude vs time
    rpy_plot = Plot(
        title='Attitude',
        xlabel=f'Time (s) (t0 = {t0})',
        ylabels=['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'],
    )
    for channel, data in pva_data.items():
        pos_plot.add_data(
            data.label, data.time, data.llh.T, is_scatter=True, marker='.'
        )
        vel_plot.add_data(
            data.label, data.time, data.vel.T, is_scatter=True, marker='.'
        )
        if data.rpy.size:
            rpy_plot.add_data(
                data.label, data.time, data.rpy.T, is_scatter=True, marker='.'
            )
    pos_plot.plot(save_dir)
    vel_plot.plot(save_dir)
    if rpy_plot.data:
        rpy_plot.plot(save_dir)

    if len(pva_data.keys()) > 1 and len(pva_data[truth_channel].time) > 0:
        # Plot position error vs time on one plot
        shared_pos_err_plot = Plot(
            title='Position Error',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['North (m)', 'East (m)', 'Down (m)'],
        )
        # Plot velocity error vs time on one plot
        shared_vel_err_plot = Plot(
            title='Velocity Error',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['North (m/s)', 'East (m/s)', 'Down (m/s)'],
        )
        # Plot tilt error vs time on one plot
        shared_tilt_err_plot = Plot(
            title='Tilt Error',
            xlabel=f'Time (s) (t0 = {t0})',
            ylabels=['North (deg)', 'East (deg)', 'Down (deg)'],
        )
        for channel, data in pva_data.items():
            if channel == truth_channel:
                continue

            # Calc truth at solution time
            truth = interpolate_pva(data, pva_data[truth_channel])

            # Calc solution error
            pos_error = data.ned - truth.ned
            ned_sig = data.ned_sig.T
            vel_error = data.vel - truth.vel
            vel_sig = data.vel_sig.T
            tilt_error = calc_tilts(truth.rpy, data.rpy).T if data.rpy.size else None
            tilt_sig = data.tilt_sig.T if data.rpy.size else None

            # Add error to shared plot
            shared_pos_err_plot.add_data('Error', data.time, pos_error)
            shared_vel_err_plot.add_data('Error', data.time, vel_error)
            if tilt_error is not None:
                shared_tilt_err_plot.add_data('Error', data.time, tilt_error)

            # Individual position error w/ sigma plot for each channel
            pos_err_plot = Plot(
                title=f'Position Error for {data.label}',
                xlabel=f'Time (s) (t0 = {t0})',
                ylabels=['North (m)', 'East (m)', 'Down (m)'],
                legend=['Error', '+/- 1 sigma'],
            )
            pos_err_plot.add_data(
                'Error', data.time, pos_error, is_scatter=True, marker='.'
            )
            pos_err_plot.add_data(
                'Error', data.time, ned_sig, color='black', linestyle='--'
            )
            pos_err_plot.add_data(
                'Error', data.time, -ned_sig, color='black', linestyle='--'
            )
            pos_err_plot.plot(save_dir)

            # Individual velocity error w/ sigma plot for each channel
            vel_err_plot = Plot(
                title=f'Velocity Error for {data.label}',
                xlabel=f'Time (s) (t0 = {t0})',
                ylabels=['North (m/s)', 'East (m/s)', 'Down (m/s)'],
                legend=['Error', '+/- 1 sigma'],
            )
            vel_err_plot.add_data(
                'Error', data.time, vel_error, is_scatter=True, marker='.'
            )
            vel_err_plot.add_data(
                'Error', data.time, vel_sig, color='black', linestyle='--'
            )
            vel_err_plot.add_data(
                'Error', data.time, -vel_sig, color='black', linestyle='--'
            )
            vel_err_plot.plot(save_dir)

            if tilt_error is not None:
                # Individual tilt error w/ sigma plot for each channel
                tilt_err_plot = Plot(
                    title=f'Tilt Error for {data.label}',
                    xlabel=f'Time (s) (t0 = {t0})',
                    ylabels=['North (deg)', 'East (deg)', 'Down (deg)'],
                    legend=['Error', '+/- 1 sigma'],
                )
                tilt_err_plot.add_data(
                    'Error', data.time, tilt_error, is_scatter=True, marker='.'
                )
                tilt_err_plot.add_data(
                    'Error', data.time, tilt_sig, color='black', linestyle='--'
                )
                tilt_err_plot.add_data(
                    'Error', data.time, -tilt_sig, color='black', linestyle='--'
                )
                tilt_err_plot.plot(save_dir)

        if len(shared_pos_err_plot.data) >= 2:
            shared_pos_err_plot.plot(save_dir)
            shared_vel_err_plot.plot(save_dir)
            if shared_tilt_err_plot.data:
                shared_tilt_err_plot.plot(save_dir)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="""Plot PVA messages from LCM log.""")

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all PVA messages in log. If not set, will prompt user to determine which PVA channels should be plotted.',
        action='store_true',
    )
    parser.add_argument(
        '-t',
        '--truth',
        default=None,
        help='Channel to use as truth. Will default to whatever is stored in sensors.toml config file.',
    )
    args = parser.parse_args()

    log_data = read_pva(args.logfile, args.all, args.truth)
    plot_pva(log_data)


if __name__ == '__main__':
    main()
