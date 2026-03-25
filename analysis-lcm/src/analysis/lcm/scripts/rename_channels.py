#!/usr/bin/env python3
import argparse
import os

from lcm import EventLog
from tabulate import tabulate

CHANNELS_TO_RENAME = {
    '/sensor/ublox-neo-m9n-update/geodetic_pos': '/sensor/Ublox-NEO-M9N/position',
    '/solution/ins-d/pva': '/sensor/ins-d/pva',
}


def rename_channels(args: argparse.Namespace):
    read_log = EventLog(args.filepath, 'r')

    basename, ext = os.path.splitext(args.filepath)
    out_filename = f'{basename}_renamed{ext}'
    write_log = EventLog(out_filename, 'w', True)

    msg = read_log.read_next_event()

    print('Renaming the following channels:')
    table_data = list(CHANNELS_TO_RENAME.items())
    print(tabulate(table_data, headers=['Old', 'New'], tablefmt='grid'))

    while msg is not None:
        if msg.channel in CHANNELS_TO_RENAME:
            write_log.write_event(
                msg.timestamp, CHANNELS_TO_RENAME[msg.channel], msg.data
            )
        else:
            write_log.write_event(msg.timestamp, msg.channel, msg.data)

        msg = read_log.read_next_event()

    print(f'Output log saved to {out_filename}')
    read_log.close()
    write_log.close()


def main():
    parser = argparse.ArgumentParser(
        description="Rename channels in LCM log. Outputs new log to '<old_log_basename>_renamed.<ext>'"
    )
    parser.add_argument('filepath', help='Full path to LCM Log file', type=str)
    args = parser.parse_args()
    rename_channels(args)


if __name__ == '__main__':
    main()
