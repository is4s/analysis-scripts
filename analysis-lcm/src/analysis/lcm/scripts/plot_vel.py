#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import LogData, VelData
from analysis.lcm.log_readers import read_vel


def plot_vel(log_data: LogData[VelData]) -> None:
    vel_data = log_data.data
    t0 = log_data.t0
    truth_channel = log_data.truth_channel

    plt.figure('Body Velocity')
    plt.suptitle('Body Velocity')
    plt.gca().remove()
    for chan, data in vel_data.items():
        plt.subplot(3, 1, 1)
        plt.plot(data.time, data.vel[:, 0], label=chan)
        plt.subplot(3, 1, 2)
        plt.plot(data.time, data.vel[:, 1], label=chan)
        plt.subplot(3, 1, 3)
        plt.plot(data.time, data.vel[:, 2], label=chan)
        print(chan, np.mean(data.vel[:, 1]))
    ax1 = plt.subplot(3, 1, 1)
    plt.ylabel('Forward (m/s)')
    ax2 = plt.subplot(3, 1, 2)
    plt.ylabel('Lateral (m/s)')
    ax3 = plt.subplot(3, 1, 3)
    plt.ylabel('Vertical (m/s)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Time (s), t0 = {t0}')
    plt.legend()
    plt.tight_layout()

    if truth_channel is not None and len(vel_data[truth_channel].time) > 0:
        # Plot velocity error vs time
        plt.figure('Velocity Error')
        plt.suptitle('Velocity Error')
        plt.gca().remove()
        for channel, data in vel_data.items():
            if channel == truth_channel:
                continue

            truth_data = vel_data[truth_channel]
            truth_x = np.interp(data.time, truth_data.time, truth_data.vel[:, 0])
            truth_y = np.interp(data.time, truth_data.time, truth_data.vel[:, 1])
            truth_z = np.interp(data.time, truth_data.time, truth_data.vel[:, 2])
            plt.subplot(3, 1, 1)
            plt.scatter(data.time, data.vel[:, 0] - truth_x, label=channel, marker='.')
            plt.subplot(3, 1, 2)
            plt.scatter(data.time, data.vel[:, 1] - truth_y, label=channel, marker='.')
            plt.subplot(3, 1, 3)
            plt.scatter(data.time, data.vel[:, 2] - truth_z, label=channel, marker='.')
        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel('Forward (m/s)')
        ax2 = plt.subplot(3, 1, 2)
        plt.ylabel('Lateral (m/s)')
        ax3 = plt.subplot(3, 1, 3)
        plt.ylabel('Vertical (m/s)')
        ax1.sharex(ax2)
        ax2.sharex(ax3)
        plt.xlabel(f'Time (s), t0 = {t0}')
        plt.legend()
        plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="""Plot velocity messages from LCM log."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all velocity messages in log. If not set, will prompt user to determine which velocity channels should be plotted.',
        action='store_true',
    )

    args = parser.parse_args()
    log_data = read_vel(args.logfile, args.all)
    plot_vel(log_data)


if __name__ == '__main__':
    main()
