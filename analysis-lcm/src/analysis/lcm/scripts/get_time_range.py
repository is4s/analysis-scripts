#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from analysis.lcm.measurements import decode_aspn_lcm_msg
from aspn23_lcm import measurement_time
from aspn23_xtensor import TypeTimestamp
from lcm import Event, EventLog
from tqdm import tqdm

# Map common abbreviations to IANA zone names
TZ_ABBREVIATIONS = {
    'UTC': 'UTC',
    'PST': 'America/Los_Angeles',
    'PDT': 'America/Los_Angeles',
    'MST': 'America/Denver',
    'MDT': 'America/Denver',
    'CST': 'America/Chicago',
    'CDT': 'America/Chicago',
    'EST': 'America/New_York',
    'EDT': 'America/New_York',
    'HST': 'Pacific/Honolulu',
    'AKST': 'America/Anchorage',
    'AKDT': 'America/Anchorage',
}


def parse_timezone(tzstr: str) -> ZoneInfo:
    tzstr = tzstr.upper()
    if tzstr in TZ_ABBREVIATIONS:
        tzname = TZ_ABBREVIATIONS[tzstr]
    else:
        tzname = tzstr  # allow full names too (e.g. "Europe/London")
    try:
        return ZoneInfo(tzname)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid timezone '{tzstr}': {e}")


def get_time_range(logfile: str, tz: ZoneInfo) -> None:
    log = EventLog(logfile, 'r')

    times: dict[str, list[TypeTimestamp]] = {}

    print('Reading measurements from input log...')
    channels_to_ignore = set()
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

        time, data = decode_aspn_lcm_msg(msg)
        if not isinstance(data, measurement_time):
            channels_to_ignore.add(msg.channel)
            continue

        if msg.channel not in times:
            times[msg.channel] = []
            print(f'Found channel {msg.channel}')

        times[msg.channel].append(data.elapsed_nsec[0] / 1e9)  # type: ignore[union-attr]

    for channel, time in times.items():
        start_time = datetime.fromtimestamp(time[0], tz=tz)
        end_time = datetime.fromtimestamp(time[-1], tz=tz)
        print(f'{channel}:')
        print(f'\tStart: {start_time:%D %H:%M:%S}')
        print(f'\tEnd: {end_time:%D %H:%M:%S}')


def main():
    parser = argparse.ArgumentParser(
        description='Print datetime span of UTC time measurements in log, in the selected time zone.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('logfile', help='Full path to LCM Log file', type=str)
    parser.add_argument(
        '--tz',
        type=parse_timezone,
        default=ZoneInfo('UTC'),
        help=(
            'Timezone abbreviation. Available abbreviations:\n'
            '\t' + '\n\t'.join(f'{key}: {val}' for key, val in TZ_ABBREVIATIONS.items())
        ),
    )
    args = parser.parse_args()
    get_time_range(args.logfile, args.tz)


if __name__ == '__main__':
    main()
