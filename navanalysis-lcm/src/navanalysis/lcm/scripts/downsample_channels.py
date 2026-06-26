#!/usr/bin/env python3
import os
import sys

from lcm import EventLog


class Downsample:
    factor: int
    cur_idx: int

    def __init__(self, factor):
        # Keep 1 out of every <factor> measurements
        self.factor = factor
        self.cur_idx = -1


CHANNELS_TO_DOWNSAMPLE: dict[str, Downsample] = {
    'channel_to_downsample': Downsample(100)
}


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 downsample_channels.py <logfile>')
        exit()

    in_filename = sys.argv[1]
    read_log = EventLog(in_filename, 'r')

    out_filename = os.path.splitext(in_filename)[0] + '_downsampled.lcm'
    write_log = EventLog(out_filename, 'w', True)

    for msg in read_log:
        if msg.channel in CHANNELS_TO_DOWNSAMPLE:
            downsample = CHANNELS_TO_DOWNSAMPLE[msg.channel]
            downsample.cur_idx = (downsample.cur_idx + 1) % downsample.factor
            if downsample.cur_idx > 0:
                # Drop measurement
                continue

        write_log.write_event(msg.timestamp, msg.channel, msg.data)

    read_log.close()
    write_log.close()


if __name__ == '__main__':
    main()
