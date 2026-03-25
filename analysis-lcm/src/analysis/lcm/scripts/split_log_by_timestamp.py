#!/usr/bin/env python3
"""Split an LCM log around a given timestamp, with all measurements before the
timestamp being saved to one log, and all measurements after the timestamp
saved to another log."""

import argparse
import os

from analysis.lcm.measurements import decode_aspn_lcm_msg
from aspn23_xtensor import TypeTimestamp, to_type_timestamp
from lcm import Event, EventLog


# Takes in LCM logfile and writes a new logfile with a modified channel
def main(logfile: str, split_time: TypeTimestamp):
    read_log = EventLog(logfile, 'r')

    out_filename1 = os.path.splitext(logfile)[0] + '1.lcm'
    out_filename2 = os.path.splitext(logfile)[0] + '2.lcm'
    write_log1 = EventLog(out_filename1, 'w', True)
    write_log2 = EventLog(out_filename2, 'w', True)

    msg: Event
    for msg in read_log:
        t, aspn_msg = decode_aspn_lcm_msg(msg)
        assert t is not None

        if t < split_time:
            write_log1.write_event(msg.timestamp, msg.channel, msg.data)
        else:
            write_log2.write_event(msg.timestamp, msg.channel, msg.data)

        msg = read_log.read_next_event()

    read_log.close()
    write_log1.close()
    write_log2.close()

    print(f'Log 1 saved to {out_filename1}.')
    print(f'Log 2 saved to {out_filename2}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Split an LCM log around a given timestamp, with all
                    measurements before the timestamp being saved to one log,
                    and all measurements after the timestamp saved to another
                    log."""
    )
    parser.add_argument('filepath', help='Full path to LCM Log file.', type=str)
    parser.add_argument(
        'split_time',
        help='Time in seconds around which to split the log.',
        type=float,
    )

    args = parser.parse_args()
    main(args.filepath, to_type_timestamp(args.split_time))
