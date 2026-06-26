#!/usr/bin/env python3
import sys

from lcm import Event, EventLog


# Takes in LCM logfile and prints the channels found in the file
def print_channels(logfile: str):
    read_log = EventLog(logfile, 'r')

    channels = []
    msg: Event
    for msg in read_log:
        if not msg.channel in channels:
            channels.append(msg.channel)
    read_log.close()

    print(f'Channels in {logfile}:')
    channels.sort()
    for channel in channels:
        print(f'\t{channel}')


def main():
    if len(sys.argv) != 2:
        print('Please provide an lcm log file.')
        exit()

    filename = sys.argv[1]
    print_channels(filename)


if __name__ == '__main__':
    main()
