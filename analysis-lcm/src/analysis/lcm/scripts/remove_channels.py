#!/usr/bin/env python3
import os
import sys

from lcm import Event, EventLog


def remove_channels(in_filename: str):
    read_log = EventLog(in_filename, 'r')

    basename, ext = os.path.splitext(in_filename)
    out_filename = f'{basename}_trimmed{ext}'
    write_log = EventLog(out_filename, 'w', True)

    channels_to_keep = set()
    channels_to_remove = set()
    msg: Event
    for msg in read_log:
        if msg.channel in channels_to_keep:
            write_log.write_event(msg.timestamp, msg.channel, msg.data)
        elif msg.channel not in channels_to_remove:
            # Get user input to decide whether to keep this channel
            usr_input = input(f'Found channel {msg.channel}. Keep? [y/n]')
            if usr_input == 'y':
                channels_to_keep.add(msg.channel)
                write_log.write_event(msg.timestamp, msg.channel, msg.data)
            else:
                channels_to_remove.add(msg.channel)

    read_log.close()
    write_log.close()

    # Print summary
    print(f'Kept channels:')
    for channel in channels_to_keep:
        print(f'\t{channel}')
    print()
    print(f'Removed channels:')
    for channel in channels_to_remove:
        print(f'\t{channel}')

    print(f'Trimmed log saved to {out_filename}.')


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 remove_channels.py <logfile>')
        exit()

    remove_channels(sys.argv[1])
