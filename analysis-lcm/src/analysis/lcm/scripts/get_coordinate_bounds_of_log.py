import sys

import numpy as np
from analysis.lcm.measurements import decode_aspn_lcm_msg, get_pos
from lcm import Event, EventLog


def main(logfile, channel):
    log = EventLog(logfile, 'r')

    min_lat = min_lon = max_lat = max_lon = 0

    msg: Event
    for msg in log:
        if msg.channel == '/solution/ins-d/pva':
            t, aspn_msg = decode_aspn_lcm_msg(msg)
            llh, _ = get_pos(aspn_msg)
            if not min_lat or llh[0] < min_lat:
                min_lat = llh[0]
            if not max_lat or llh[0] > max_lat:
                max_lat = llh[0]
            if not min_lon or llh[1] < min_lon:
                min_lon = llh[1]
            if not max_lon or llh[1] > max_lon:
                max_lon = llh[1]

    log.close()
    print('Lat bounds: [%f, %f]' % (np.rad2deg(min_lat), np.rad2deg(max_lat)))
    print('Lon bounds: [%f, %f]' % (np.rad2deg(min_lon), np.rad2deg(max_lon)))


if __name__ == '__main__':
    logfile = sys.argv[1]
    main(logfile)
