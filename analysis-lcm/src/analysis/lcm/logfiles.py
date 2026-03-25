import os
import time

from aspn23_xtensor import TypeTimestamp, to_seconds
from lcm import Event, EventLog

from .measurements import Aspn23Measurement, decode_aspn_lcm_msg


def sort_log(logfile: str) -> None:
    log = EventLog(logfile, 'r')

    start = time.time()

    # Collect all measurements into sorted list
    measurements: list[
        tuple[TypeTimestamp, str, int, Aspn23Measurement]
    ] = []

    print('Reading measurements from input log...')
    msg: Event
    for msg in log:
        t, aspn_msg = decode_aspn_lcm_msg(msg)
        if t is not None:
            x = (t, msg.channel, msg.timestamp, aspn_msg)
            measurements.append(x)

    # Sort measurements by timestamp
    print('Sorting measurements...')
    measurements.sort(key=lambda x: x[0])

    # Inject sorted measurements into new log with revised LCM timestamps
    print('Saving sorted measurements to new log...')
    basename, ext = os.path.splitext(logfile)
    out_filename = f'{basename}_sorted{ext}'
    write_log = EventLog(out_filename, 'w', True)

    lcm_t0 = measurements[0][2]  # LCM timestamp of earliest msg
    aspn_t0 = measurements[0][0]  # ASPN timestamp of earliest msg

    for cur_aspn_time, channel, old_lcm_time, aspn_msg in measurements:
        # Get cur time in microseconds for LCM timestamp
        delta_since_t0 = to_seconds(cur_aspn_time - aspn_t0)
        cur_time = round(lcm_t0 + delta_since_t0 * 1e6)
        write_log.write_event(cur_time, channel, aspn_msg.encode())

    print(f'Sorted log saved to {out_filename}')
    end = time.time()
    print(f'Runtime: {end - start:.2f} seconds')
