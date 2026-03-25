#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import LogData, SpeedData
from analysis.lcm.log_readers import read_speed


def filter_speed(data: SpeedData) -> SpeedData:
    mean_dt = np.mean(np.diff(data.time))
    N = round(1 / mean_dt)
    out = SpeedData(data.label)
    for i, speed in enumerate(data.speed):
        # Discard spikes down to ~0
        if -0.4 < speed < 0.4:
            # If speed is low, compare to speeds over last second to see if current speed is an outlier
            previous_n = data.speed[:i] if i < N else data.speed[i - N : i]

            if np.any(np.abs(previous_n) > 1):
                continue
        out.time.append(data.time[i])
        out.speed.append(speed)

    return out


def plot_speed(log_data: LogData[SpeedData]) -> None:
    speed_data = log_data.data
    truth_channel = log_data.truth_channel

    plt.figure('Speed')
    plt.title('Speed')
    for key, data in speed_data.items():
        plt.plot(data.time, data.speed, label=key)
        print(f'Median speed: {np.median(data.speed)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()

    # Filter outliers out of speed data
    plt.figure('Filtered Speed')
    plt.title('Filtered Speed')
    for label, data in speed_data.items():
        if label != truth_channel:
            filtered_data = filter_speed(data)
            # Overwrite existing speed data with filtered data
            speed_data[label] = filtered_data

        plt.plot(speed_data[label].time, speed_data[label].speed, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()

    truth_time = speed_data[truth_channel].time
    truth_speed = speed_data[truth_channel].speed
    plt.figure(f'Corrected Speed')
    plt.title(f'Corrected Speed')
    plt.plot(truth_time, truth_speed, label=truth_channel)

    for label, data in speed_data.items():
        if label == truth_channel:
            continue

        time = data.time
        s = data.speed
        interp_truth_speed = np.interp(time, truth_time, truth_speed)
        err = s - interp_truth_speed

        plt.figure(f'Speed Error for {label}')
        plt.title(f'Speed Error for {label}')
        plt.scatter(time, err, marker='.')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m/s)')
        plt.tight_layout()

        plt.figure(f'Speed Error vs. Speed for {label}')
        plt.title(f'Speed Error vs. Speed for {label}')
        plt.scatter(
            interp_truth_speed,
            err,
            label='Error',
            # cmap='Blues_r',
            # c=truth_time,
        )
        # plot best fit line
        a, b = np.polyfit(interp_truth_speed, err, 1)
        plt.plot(
            interp_truth_speed,
            a * interp_truth_speed + b,
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Best Fit (slope={a:.3f})',
        )
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Error (m/s)')
        # plt.colorbar(label='Time (s)')
        plt.tight_layout()
        plt.legend()

        corr_speed = s / (1 + a)
        plt.figure('Corrected Speed')
        plt.plot(time, corr_speed, label=label)

        err = corr_speed - interp_truth_speed
        plt.figure(f'Corrected Speed Error for {label}')
        plt.title(f'Corrected Speed Error for {label}')
        plt.scatter(time, err, marker='.')
        plt.plot(time, np.zeros_like(time) + np.std(err), color='k', linestyle='--')
        plt.plot(time, np.zeros_like(time) - np.std(err), color='k', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m/s)')
        plt.tight_layout()

    plt.figure('Corrected Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="""Plot speed messages from LCM log."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all speed messages in log. If not set, will prompt user to determine which speed channels should be plotted.',
        action='store_true',
    )

    args = parser.parse_args()
    log_data = read_speed(args.logfile, args.all)
    plot_speed(log_data)


if __name__ == '__main__':
    main()
