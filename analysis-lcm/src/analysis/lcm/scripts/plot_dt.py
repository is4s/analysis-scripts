#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.measurements import decode_aspn_lcm_msg
from aspn23_xtensor import TypeTimestamp, to_seconds
from lcm import Event, EventLog
from tqdm import tqdm


def plot_dt(logfile: str) -> None:
    log = EventLog(logfile, 'r')

    times: dict[str, list[TypeTimestamp]] = {}
    t0 = None

    msg: Event
    log_size = log.size()
    progressbar = tqdm(total=log_size, unit='B', unit_scale=True)
    fpos = log.tell()
    for msg in log:
        new_fpos = log.tell()
        progressbar.update(new_fpos - fpos)
        fpos = new_fpos

        t, aspn_msg = decode_aspn_lcm_msg(msg)

        if t is not None:
            if msg.channel not in times:
                if t0 is None:
                    t0 = t
                times[msg.channel] = []
            times[msg.channel].append(t)

    for channel, time in times.items():
        time = np.array(time)
        rel_time = np.array([to_seconds(t - t0) for t in time])
        dt = np.diff(rel_time)

        total_dt = rel_time[-1] - rel_time[0]
        N = len(rel_time)
        avg_dt = total_dt / (N - 1)
        plt.figure(f'DT {channel}')
        plt.suptitle(f'DT {channel}\n(avg = {avg_dt:.9f}s)')
        plt.scatter(rel_time[1:], dt, marker='.', label=channel)
        plt.xlabel('Time (s)')
        plt.ylabel('DT (s)')
        print(f'Avg dt for {channel}: {avg_dt:.9f}s')

    plt.legend()
    plt.show()


def main():
    logfile = sys.argv[1]
    plot_dt(logfile)


if __name__ == '__main__':
    main()
