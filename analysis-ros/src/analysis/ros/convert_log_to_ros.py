import argparse
import os
import shutil
from contextlib import suppress

import rosbag2_py
from aspn23_lcm import LcmMsg, from_lcm_map
from aspn23_ros_utils.ros_translations import to_ros_map
from lcm import Event, EventLog
from rclpy.serialization import serialize_message

from analysis.lcm.measurements import decode_aspn_lcm_msg


def convert_log_to_ros(logfile: str) -> None:
    log = EventLog(logfile)
    bag_path, _ = os.path.splitext(args.filepath)
    with suppress(FileNotFoundError):
        shutil.rmtree(bag_path)

    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('cdr', 'cdr')
    writer.open(storage_options, converter_options)

    registered_topics: set[str] = set()
    msg: Event
    for msg in log:  # type: ignore
        _, lcm_msg = decode_aspn_lcm_msg(msg)
        assert isinstance(lcm_msg, LcmMsg)  # Make type checker happy
        aspn_msg = from_lcm_map[type(lcm_msg)](lcm_msg)
        ros_msg = to_ros_map[type(aspn_msg)](aspn_msg)

        topic = msg.channel.replace('-', '_')
        if topic not in registered_topics:
            if '-' in msg.channel:
                print(f'WARNING: {msg.channel} has dashes, changing to underscores.')
            msg_type_str = '/'.join(
                type(ros_msg).__module__.split('.')[:-1] + [type(ros_msg).__name__]
            )
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    name=topic,
                    type=msg_type_str,
                    serialization_format='cdr',
                )
            )
            registered_topics.add(topic)

        writer.write(topic, serialize_message(ros_msg), msg.timestamp * 1000)

    log.close()
    writer.close()
    print(f'ROS bag written to {bag_path}/.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ASPN23 LCM log to ROS2 bag.')
    parser.add_argument('filepath', help='Full path to LCM Log file', type=str)
    args = parser.parse_args()
    convert_log_to_ros(args.filepath)
