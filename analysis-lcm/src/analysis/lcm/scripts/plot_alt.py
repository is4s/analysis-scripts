#!/usr/bin/env python3

#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import AltData, LogData
from analysis.lcm.log_readers import read_alt


def plot_alt(log_data: LogData[AltData]):
    alt_data = log_data.data
    truth_channel = log_data.truth_channel

    plt.figure('Altitude')
    plt.suptitle('Altitude')
    for channel in alt_data:
        # Plot altitude over time
        plt.scatter(
            alt_data[channel].time,
            alt_data[channel].alt,
            label=channel,
            marker='.',
        )

    plt.xlabel('Time (s)')
    plt.ylabel('Alt (m)')
    plt.legend()

    if len(alt_data.keys()) > 1 and len(alt_data[truth_channel].time) > 0:
        # Plot altitude error vs time
        plt.figure('Altitude Error')
        plt.suptitle('Altitude Error')
        plt.gca().remove()
        for channel, data in alt_data.items():
            if channel == truth_channel:
                continue

            # Altitude errors all on one plot
            plt.figure('Altitude Error')
            truth_data = alt_data[truth_channel]
            # data.time -= 2.5
            truth_alt = np.interp(data.time, truth_data.time, truth_data.alt)
            error = data.alt - truth_alt
            plt.scatter(data.time, error, label=channel, marker='.')

            init_bias = error[0]
            data.alt -= init_bias
            error = data.alt - truth_alt

            # Individual temp plot for each channel, if temp is available
            if len(data.temp) > 0:
                plt.figure(f'Temperature for {channel}')
                plt.suptitle(f'Temperature for {channel}')
                plt.scatter(data.temp_time, data.temp, marker='.')
                plt.ylabel('Temperature (K)')
                plt.xlabel('Time (s)')
                plt.tight_layout()

            # Plot altitude error as a function of truth altitude
            plt.figure(f'Altitude Error vs Altitude for {channel}')
            plt.suptitle(
                f'Altitude Error vs Altitude for {channel}\n(Initial Bias = {init_bias:.2f} m)'
            )
            plt.scatter(
                truth_alt,
                error,
                label='Error',
                marker='.',
            )
            a, b = np.polyfit(truth_alt, error, 1)
            plt.plot(
                truth_alt,
                a * truth_alt + b,
                color='orange',
                linestyle='--',
                linewidth=2,
                label=f'Best Fit (slope={a:.3f})',
            )
            plt.xlabel('Altitude (m)')
            plt.ylabel('Error (m)')
            plt.tight_layout()
            plt.legend()

            corr_alt = data.alt / (1 + a) - b  # TODO: is this math right?
            corr_error = corr_alt - truth_alt

            # Individual altitude plot for each channel
            plt.figure(f'Altitude for {channel}')
            plt.suptitle(f'Altitude for {channel}\n(Initial Bias = {init_bias:.2f} m)')
            plt.scatter(data.time, data.alt + init_bias, label='raw', marker='.')
            plt.scatter(data.time, data.alt, label='bias removed', marker='.')
            plt.scatter(data.time, corr_alt, label='scale factor removed', marker='.')
            plt.scatter(
                truth_data.time,
                truth_data.alt,
                label=truth_channel,
                marker='.',
            )
            plt.ylabel('Error (m)')
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.legend()

            # Individual altitude error plot for each channel
            plt.figure(f'Altitude Error for {channel}')
            plt.suptitle(
                f'Altitude Error for {channel}\n(Initial Bias = {init_bias:.2f} m)'
            )
            plt.scatter(data.time, error + init_bias, label='raw', marker='.')
            plt.scatter(data.time, error, label='bias removed', marker='.')
            plt.scatter(data.time, corr_error, label='scale factor removed', marker='.')
            plt.ylabel('Error (m)')
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.legend()

        plt.figure('Altitude Error')
        plt.ylabel('Error (m)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="""Plot altitude messages from LCM log."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all altitude messages in log. If not set, will prompt user to determine which altitude channels should be plotted.',
        action='store_true',
    )

    args = parser.parse_args()
    log_data = read_alt(args.logfile, args.all)
    plot_alt(log_data)


if __name__ == '__main__':
    main()
