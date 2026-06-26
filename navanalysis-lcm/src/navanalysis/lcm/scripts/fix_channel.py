import argparse
import os

import numpy as np
from aspn23_lcm import measurement_position_velocity_attitude
from lcm import Event, EventLog
from navtk.navutils import quat_to_rpy, rpy_to_quat


def fix_channel(args: argparse.Namespace) -> None:
    basename, ext = os.path.splitext(args.filepath)
    log = EventLog(args.filepath)
    out_filename = f'{basename}_mod{ext}'
    out_log = EventLog(out_filename, 'w', True)

    msg: Event
    for msg in log:
        fp = msg.data[:8]

        if msg.channel == '/sensor/ins-d/pva':
            data = measurement_position_velocity_attitude.decode(msg.data)
            rpy = quat_to_rpy(data.quaternion)
            dcm = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            new_rpy = dcm @ rpy
            new_quat = rpy_to_quat(new_rpy)
            data.quaternion = new_quat

            data.v3 = -data.v3

            out_log.write_event(msg.timestamp, msg.channel, data.encode())
        else:
            out_log.write_event(msg.timestamp, msg.channel, msg.data)

    print(f'Modified log saved to {out_filename}')


def main():
    parser = argparse.ArgumentParser(
        description="Modifies log in some way. Outputs modified log to '<old_log_basename>_mod.<ext>'"
    )
    parser.add_argument('filepath', help='Full path to LCM Log file', type=str)
    args = parser.parse_args()
    fix_channel(args)


if __name__ == '__main__':
    main()
