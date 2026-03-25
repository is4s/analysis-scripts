#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.measurements import decode_aspn_lcm_msg
from aspn23_xtensor import TypeTimestamp, to_seconds
from lcm import Event, EventLog
from tqdm import tqdm


def plot_time(logfile: str, extract_all: bool) -> None:
    log = EventLog(logfile, 'r')

    times: dict[str, list[TypeTimestamp]] = {}
    t0 = None

    channels_to_ignore = set()
    print('Reading measurements from input log...')
    msg: Event
    log_size = log.size()
    progressbar = tqdm(total=log_size, unit='B', unit_scale=True)
    fpos = log.tell()
    for msg in log:
        new_fpos = log.tell()
        progressbar.update(new_fpos - fpos)
        fpos = new_fpos

        if msg.channel in channels_to_ignore:
            continue

        # Determine if we want to plot timestamps from this channel
        if msg.channel not in times:
            if extract_all:
                times[msg.channel] = []
            elif input(f'Found channel {msg.channel}. Plot time? [y/n]') == 'y':
                times[msg.channel] = []
            else:
                channels_to_ignore.add(msg.channel)
                continue  # skip this message

        time, data = decode_aspn_lcm_msg(msg)
        if time is None:
            continue

        if t0 is None:
            t0 = time

        times[msg.channel].append(time)  # type: ignore[union-attr]

    plt.figure('Time', figsize=(10, 8))
    plt.suptitle('Time')
    idx = 0
    for channel, time in times.items():
        time = np.array([to_seconds(t - t0) for t in time])
        plt.scatter(time, np.zeros_like(time) + idx, label=f'{channel} (y={idx})')
        idx += 1
    plt.xlabel(f'Time (s) (t0={t0})')
    plt.ylabel('Channel Index')
    plt.legend(
        fontsize=10,
        reverse=True,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        # borderaxespad=0.0,
    )
    plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot timestamps of the ASPN messages in an LCM log.'
    )
    parser.add_argument('logfile', help='Full path to LCM Log file', type=str)
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all timestamps in log. If not set, will prompt user to determine which channels should be plotted.',
        action='store_true',
    )
    args = parser.parse_args()
    plot_time(args.logfile, args.all)


if __name__ == '__main__':
    main()
