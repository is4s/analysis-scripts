#!/usr/bin/env python3

import warnings
from typing import Union

import aspn23
import aspn23_lcm
import numpy as np
from aspn23_xtensor import TypeTimestamp, to_type_timestamp
from lcm import Event
from navtk.navutils import (
    quat_to_dcm,
    quat_to_rpy,
    rpy_to_dcm,
)

from .conversions import pressure_to_alt

BILLION = 1_000_000_000


Aspn23Measurement = Union[
    aspn23_lcm.measurement_angular_velocity_1d,
    aspn23_lcm.measurement_angular_velocity,
    aspn23_lcm.measurement_accumulated_distance_traveled,
    aspn23_lcm.measurement_altitude,
    aspn23_lcm.measurement_attitude_2d,
    aspn23_lcm.measurement_attitude_3d,
    aspn23_lcm.measurement_barometer,
    aspn23_lcm.measurement_delta_position,
    aspn23_lcm.measurement_delta_range,
    aspn23_lcm.measurement_delta_range_to_point,
    aspn23_lcm.measurement_direction_2d_to_points,
    aspn23_lcm.measurement_direction_3d_to_points,
    aspn23_lcm.measurement_direction_of_motion_2d,
    aspn23_lcm.measurement_direction_of_motion_3d,
    aspn23_lcm.measurement_frequency_difference,
    aspn23_lcm.measurement_heading,
    aspn23_lcm.measurement_image,
    aspn23_lcm.measurement_IMU,
    aspn23_lcm.measurement_magnetic_field,
    aspn23_lcm.measurement_magnetic_field_magnitude,
    aspn23_lcm.measurement_position,
    aspn23_lcm.measurement_position_attitude,
    aspn23_lcm.measurement_position_velocity_attitude,
    aspn23_lcm.measurement_range_rate_to_point,
    aspn23_lcm.measurement_range_to_point,
    aspn23_lcm.measurement_satnav,
    aspn23_lcm.measurement_satnav_subframe,
    aspn23_lcm.measurement_satnav_with_sv_data,
    aspn23_lcm.measurement_specific_force_1d,
    aspn23_lcm.measurement_speed,
    aspn23_lcm.measurement_TDOA_1Tx_2Rx,
    aspn23_lcm.measurement_TDOA_2Tx_1Rx,
    aspn23_lcm.measurement_temperature,
    aspn23_lcm.measurement_time,
    aspn23_lcm.measurement_time_difference,
    aspn23_lcm.measurement_time_frequency_difference,
    aspn23_lcm.measurement_velocity,
]


ASPN23_MEASUREMENTS: list[type[Aspn23Measurement]] = [
    aspn23_lcm.measurement_angular_velocity_1d,
    aspn23_lcm.measurement_angular_velocity,
    aspn23_lcm.measurement_accumulated_distance_traveled,
    aspn23_lcm.measurement_altitude,
    aspn23_lcm.measurement_attitude_2d,
    aspn23_lcm.measurement_attitude_3d,
    aspn23_lcm.measurement_barometer,
    aspn23_lcm.measurement_delta_position,
    aspn23_lcm.measurement_delta_range,
    aspn23_lcm.measurement_delta_range_to_point,
    aspn23_lcm.measurement_direction_2d_to_points,
    aspn23_lcm.measurement_direction_3d_to_points,
    aspn23_lcm.measurement_direction_of_motion_2d,
    aspn23_lcm.measurement_direction_of_motion_3d,
    aspn23_lcm.measurement_frequency_difference,
    aspn23_lcm.measurement_heading,
    aspn23_lcm.measurement_image,
    aspn23_lcm.measurement_IMU,
    aspn23_lcm.measurement_magnetic_field,
    aspn23_lcm.measurement_magnetic_field_magnitude,
    aspn23_lcm.measurement_position,
    aspn23_lcm.measurement_position_attitude,
    aspn23_lcm.measurement_position_velocity_attitude,
    aspn23_lcm.measurement_range_rate_to_point,
    aspn23_lcm.measurement_range_to_point,
    aspn23_lcm.measurement_satnav,
    aspn23_lcm.measurement_satnav_subframe,
    aspn23_lcm.measurement_satnav_with_sv_data,
    aspn23_lcm.measurement_specific_force_1d,
    aspn23_lcm.measurement_speed,
    aspn23_lcm.measurement_TDOA_1Tx_2Rx,
    aspn23_lcm.measurement_TDOA_2Tx_1Rx,
    aspn23_lcm.measurement_temperature,
    aspn23_lcm.measurement_time,
    aspn23_lcm.measurement_time_difference,
    aspn23_lcm.measurement_time_frequency_difference,
    aspn23_lcm.measurement_velocity,
]


def get_aspn23_time(aspn23_msg: Aspn23Measurement):
    return TypeTimestamp(aspn23_msg.time_of_validity.elapsed_nsec)


def get_pva(aspn_msg):
    llh = None
    vel = None
    rpy = None
    ned_sig = None
    vel_sig = None
    tilt_sig = None
    if isinstance(
        aspn_msg, aspn23_lcm.measurement_position_velocity_attitude
    ) or isinstance(aspn_msg, aspn23.MeasurementPositionVelocityAttitude):
        llh = np.array(
            [
                aspn_msg.p1,
                aspn_msg.p2,
                aspn_msg.p3,
            ]
        )
        vel = np.array(
            [
                aspn_msg.v1,
                aspn_msg.v2,
                aspn_msg.v3,
            ]
        )
        rpy = quat_to_rpy(aspn_msg.quaternion)
        sig = np.sqrt(np.diag(aspn_msg.covariance))
        ned_sig = sig[:3]
        vel_sig = sig[3:6]
        tilt_sig = sig[6:]

    return llh, vel, rpy, ned_sig, vel_sig, tilt_sig


def get_pos(aspn_msg):
    pos = None
    sig = None
    if isinstance(aspn_msg, aspn23_lcm.measurement_position):
        pos = np.array([aspn_msg.term1, aspn_msg.term2, aspn_msg.term3])
        sig = np.sqrt(np.diag(aspn_msg.covariance))
    elif isinstance(aspn_msg, aspn23_lcm.measurement_position_velocity_attitude):
        pos = np.array([aspn_msg.p1, aspn_msg.p2, aspn_msg.p3])
        if aspn_msg.covariance:
            sig = np.sqrt(np.diag(aspn_msg.covariance)[:3])

    return pos, sig


def get_altitude(aspn_msg, bias=0, temp=288.15, ref_pressure=101325.0, ref_alt=0.0):
    alt = None
    sig = None

    if isinstance(aspn_msg, aspn23_lcm.measurement_barometer):
        alt = pressure_to_alt(aspn_msg.pressure, temp, ref_pressure, ref_alt) - bias
        sig = np.sqrt(aspn_msg.variance)
    elif isinstance(aspn_msg, aspn23_lcm.measurement_position):
        alt = aspn_msg.term3 - bias
        sig = np.sqrt(aspn_msg.covariance[2][2])
    elif isinstance(aspn_msg, aspn23_lcm.measurement_position_velocity_attitude):
        alt = aspn_msg.p3 - bias
        if aspn_msg.covariance:
            sig = np.sqrt(aspn_msg.covariance[2][2])
    elif isinstance(aspn_msg, aspn23_lcm.measurement_altitude):
        alt = aspn_msg.altitude - bias
        sig = np.sqrt(aspn_msg.variance)

    return alt, sig


def get_mag(aspn_msg):
    mag = None
    sig = None

    if isinstance(aspn_msg, aspn23_lcm.measurement_magnetic_field):
        mag = np.array(
            [
                aspn_msg.x_field_strength,
                aspn_msg.y_field_strength,
                aspn_msg.z_field_strength,
            ]
        )
        sig = np.sqrt(np.diag(aspn_msg.covariance))

    return mag, sig


def get_heading(aspn_msg):
    heading = None
    sig = None
    if isinstance(aspn_msg, aspn23_lcm.measurement_position_velocity_attitude):
        rpy = quat_to_rpy(aspn_msg.quaternion)
        heading = np.rad2deg(rpy[2])
        sig = np.rad2deg(np.sqrt(aspn_msg.covariance[8][8]))

    return heading, sig


def get_imu(aspn_msg):
    accel = None
    gyro = None
    if isinstance(aspn_msg, aspn23_lcm.measurement_IMU):
        accel = np.array(aspn_msg.meas_accel)
        gyro = np.array(aspn_msg.meas_gyro)

    return accel, gyro


def get_speed(aspn_msg):
    speed = None
    sig = None

    if isinstance(aspn_msg, aspn23_lcm.measurement_position_velocity_attitude):
        speed = np.linalg.norm([aspn_msg.v1, aspn_msg.v2, aspn_msg.v3])
        sig = np.linalg.norm(np.sqrt(np.diag(aspn_msg.covariance)[3:6]))  # FIXME: bad
    elif isinstance(aspn_msg, aspn23_lcm.measurement_delta_position):
        # FIXME: only uses one term
        speed = aspn_msg.term1 / aspn_msg.delta_t
        sig = np.sqrt(aspn_msg.covariance[0][0]) / aspn_msg.delta_t
    elif isinstance(aspn_msg, aspn23_lcm.measurement_speed):
        speed = aspn_msg.speed
        sig = np.sqrt(aspn_msg.variance)
    elif isinstance(aspn_msg, aspn23_lcm.measurement_velocity):
        speed = np.linalg.norm([aspn_msg.x, aspn_msg.y, aspn_msg.z])
        sig = np.linalg.norm(np.sqrt(np.diag(aspn_msg.covariance)))  # FIXME: bad

    return speed, sig


def get_vel(aspn_msg):
    """Extract velocity and sigma in sensor frame."""
    vel = None
    sig = None
    if isinstance(aspn_msg, aspn23_lcm.measurement_velocity):
        vel = np.array([aspn_msg.x, aspn_msg.y, aspn_msg.z])
        sig = np.sqrt(np.diag(aspn_msg.covariance))
    elif isinstance(aspn_msg, aspn23_lcm.measurement_position_velocity_attitude):
        C_sensor_to_ned = quat_to_dcm(aspn_msg.quaternion)
        vel_ned = np.array([aspn_msg.v1, aspn_msg.v2, aspn_msg.v3])
        vel = C_sensor_to_ned.T @ vel_ned
        sig = np.sqrt(np.diag(aspn_msg.covariance)[3:6])

    return vel, sig


def get_temp(aspn_msg: aspn23_lcm.measurement_temperature):
    temp = None
    sig = None
    if isinstance(aspn_msg, aspn23_lcm.measurement_temperature):
        temp = aspn_msg.temperature
        sig = np.sqrt(aspn_msg.variance)

    return temp, sig


def decode_aspn_lcm_msg(
    msg: Event,
) -> tuple[TypeTimestamp | None, Aspn23Measurement | None]:
    fingerprint = msg.data[:8]

    for cls in ASPN23_MEASUREMENTS:
        if fingerprint == cls._get_packed_fingerprint():
            decoded_data = cls.decode(msg.data)
            time = get_aspn23_time(decoded_data)

            return time, decoded_data

    warnings.warn(
        message=f'Non-existent decoder for channel: {msg.channel}. Ignoring message.',
        category=RuntimeWarning,
        stacklevel=2,
    )

    return None, None


def is_type(msg: Event, *types: type[Aspn23Measurement]):
    """Return true if the LCM message matches any of the given messages types.

    Args:
        msg (Event): LCM message.
        *types: One or more message types.

    Returns:
        bool: Whether the type of msg matches any of the types.
    """
    for t in types:
        if msg.data[:8] == t._get_packed_fingerprint():
            return True

    return False
