import numpy as np
from aspn23 import MeasurementPositionVelocityAttitude as Pva
from aspn23_xtensor import TypeTimestamp, to_seconds

from analysis.lcm.data import LogData, PvaData

try:
    from aspn23_ros_interfaces.msg import MeasurementPositionVelocityAttitude as RosPva
    from aspn23_ros_utils import from_ros_map  # type: ignore[import]
    from rclpy.serialization import deserialize_message  # type: ignore[import]
    from rosbag2_py import (  # type: ignore[import]
        ConverterOptions,
        SequentialReader,
        StorageOptions,
    )
    from rosidl_runtime_py.utilities import get_message  # type: ignore[import]
except ImportError as e:
    raise ImportError(
        'Is ROS installed and the ASPN-ROS environment sourced? See the ROS '
        'usage tutorial in the documentation.'
    ) from e


# TODO: move this into analysis-scripts
class RosBagReader:
    def __init__(self, bagfile: str):
        if bagfile.endswith('.db3'):
            storage_id = 'sqlite3'
        elif bagfile.endswith('.mcap'):
            storage_id = 'mcap'
        else:
            raise ValueError(f'Invalid bagfile: {bagfile}.')

        storage_options = StorageOptions(uri=bagfile, storage_id=storage_id)
        converter_options = ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        )

        self.reader = SequentialReader()
        self.reader.open(storage_options, converter_options)
        self.type_map = {
            topic.name: topic.type for topic in self.reader.get_all_topics_and_types()
        }

    def harvest_topics(self, topics: list[str]) -> LogData[PvaData]:
        """
        Harvest messages from the list of topics.

        Assumes messages are of type MeasurementPositionVelocityAttitude.
        """
        out: LogData[PvaData] = LogData('rosbag')
        non_pva_topics = set()
        while self.reader.has_next():
            topic, data, _ = self.reader.read_next()
            if topic not in topics or topic in non_pva_topics:
                continue
            ros_msg_type = get_message(self.type_map[topic])
            ros_msg = deserialize_message(data, ros_msg_type)
            aspn_msg = from_ros_map[ros_msg_type](ros_msg)

            if not isinstance(aspn_msg, Pva):
                print(
                    f'Cannot harvest message on topic {topic}. Expected message of type {Pva} but got {type(aspn_msg)}.'
                )
                non_pva_topics.add(topic)
                continue

            t = TypeTimestamp(aspn_msg.time_of_validity.elapsed_nsec)
            if topic not in out.data:
                out.data[topic] = PvaData(topic)
                if out.t0 is None:
                    out.t0 = t

            out.data[topic].add_data(t, aspn_msg)

        for data in out.data.values():
            # Convert tov to relative time
            data.time = np.array([to_seconds(t - out.t0) for t in data.time])

            # Convert from lists to numpy arrays
            data.llh = np.array(data.llh)
            data.set_ned_pos()
            data.ned_sig = np.array(data.ned_sig)
            data.set_llh_sigma()

            data.vel = np.array(data.vel)
            data.vel_sig = np.array(data.vel_sig)

            data.rpy = np.rad2deg(np.unwrap(data.rpy, axis=0))
            data.tilt_sig = np.rad2deg(data.tilt_sig)

        return out
