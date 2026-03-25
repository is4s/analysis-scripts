#!/usr/bin/env python3

import argparse
import os
import warnings
from typing import get_args

from analysis.lcm.measurements import (
    Aspn23Measurement,
    decode_aspn_lcm_msg,
)
from lcm import Event, EventLog

SHIFTS = {
    '/sensor/xterra/speed': 7010149.651134854,
    '/sensor/xterra/steering_angle': 7010149.651134854,
    '/sensor/hall/speed': -1453.894523971,
    '/sensor/stim300/imu': -1453.894523971,
    '/sensor/hmr2300/mag_field': -1453.894523971,
    '/sensor/omron/temp': 757585,  # FIXME: probably not right
    '/sensor/omron/baro_pressure': 757585,  # FIXME: probably not right
}
BILLION = 1_000_000_000


def main(args: argparse.Namespace) -> None:
    log = EventLog(args.filepath, 'r')

    basename, ext = os.path.splitext(args.filepath)
    out_filename = f'{basename}_shifted{ext}'
    write_log = EventLog(out_filename, 'w', True)

    for shift in SHIFTS:
        print(f'Shifting channel {shift} by {SHIFTS[shift]} seconds')
        SHIFTS[shift] = int(SHIFTS[shift] * BILLION)  # convert shift to int ns

    msg: Event
    for msg in log:
        if msg.channel in SHIFTS:
            ts, aspn_msg = decode_aspn_lcm_msg(msg)
            time_offset_ns = SHIFTS[msg.channel]
            if isinstance(aspn_msg, get_args(Aspn23Measurement)):
                aspn_msg.time_of_validity.elapsed_nsec += time_offset_ns
            else:
                warnings.warn(
                    f'Cannot decode message on channel {msg.channel}. Skipping.'
                )
                continue
            write_log.write_event(msg.timestamp, msg.channel, aspn_msg.encode())
        else:
            write_log.write_event(msg.timestamp, msg.channel, msg.data)

    print(f'Shifted log written to {out_filename}')
    log.close()
    write_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Shift timestamps of specific channel in LCM log'
    )
    parser.add_argument('filepath', help='Full path to LCM Log file', type=str)
    args = parser.parse_args()
    main(args)
