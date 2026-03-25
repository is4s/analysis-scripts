#!/usr/bin/env python3

import os
import sys

import numpy as np
from analysis.lcm.measurements import decode_aspn_lcm_msg
from lcm import Event, EventLog

VARIANCES: dict[str, np.typing.NDArray[np.float64]] = {
    '/sensor/ins-d/pva': np.array([4, 4, 4, 0.4, 0.4, 0.4, 0.04, 0.04, 0.04]),
    '/sensor/ublox/position': np.array([25, 25, 25]),
    '/sensor/hmr2300/mag_field': np.zeros(3),
    '/sensor/vn-100/mag_field': np.zeros(3),
}


def set_covariance(aspn_msg, variance: np.typing.NDArray[np.float64]):
    aspn_msg.num_meas = variance.size
    aspn_msg.covariance = np.diag(variance).tolist()


def main(logfile: str) -> None:
    read_log = EventLog(logfile, 'r')

    out_filename = os.path.splitext(logfile)[0] + '_with_cov.lcm'
    write_log = EventLog(out_filename, 'w', True)

    msg: Event
    for msg in read_log:
        if msg.channel in VARIANCES:
            t, aspn_msg = decode_aspn_lcm_msg(msg)
            assert aspn_msg is not None

            set_covariance(aspn_msg, VARIANCES[msg.channel])

            write_log.write_event(msg.timestamp, msg.channel, aspn_msg.encode())
        else:
            write_log.write_event(msg.timestamp, msg.channel, msg.data)

    print(f'Output log written to {out_filename}.')


if __name__ == '__main__':
    logfile = sys.argv[1]
    main(logfile)
