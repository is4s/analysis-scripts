#!/usr/bin/env python3

"""Compare Navtk Mechanization to inu mechanization"""

import sys

import inu
import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.conversions import llh_to_ned
from analysis.lcm.interpolation import interpolate_pva_advanced
from analysis.lcm.measurements import (
    decode_aspn_lcm_msg,
    get_imu,
    get_pva,
)
from aspn23_xtensor import (
    to_seconds,
    to_type_timestamp,
)
from lcm import EventLog
from navtk.inertial import (
    DcmIntegrationMethods,
    EarthModels,
    Inertial,
    IntegrationMethods,
    MechanizationOptions,
    StandardPosVelAtt,
)
from navtk.navutils import (
    GravModels,
    calculate_gravity_savage_ned,
    calculate_gravity_schwartz,
    dcm_to_rpy,
    rpy_to_dcm,
)

# Our IMU in the sample data recorded data at 100Hz.
IMU_DT = 0.01

# Set to None to generate IMU data from PVA_CHANNEL
IMU_CHANNEL = None
IMU_CHANNEL = 'imu_channel'
C_INS_TO_PLATFORM = np.eye(3)

# Create string label for creating/calling pinson15 state block
PINSON15 = 'pinson15'

PVA_CHANNEL = '/sensor/ins-d/pva'


def schwartz(llh: np.ndarray):
    if llh.ndim == 1:
        return calculate_gravity_schwartz(llh[2], llh[0])

    out = [calculate_gravity_schwartz(alt, lat) for lat, _, alt in llh.T]

    return np.array(out).T


def savage(llh: np.ndarray):
    if llh.ndim == 1:
        return calculate_gravity_savage_ned(llh[2], llh[0])

    out = [calculate_gravity_savage_ned(alt, lat) for lat, _, alt in llh.T]

    return np.array(out).T


def navtk_mechanize(t0, imu_time, llh0, vned0, rpy0, dv_t, dth_t):
    print('Mechanizing with navtk...')

    out = np.zeros((len(imu_time) + 1, 9))
    out[0] = np.concatenate((llh0, vned0, rpy0))

    C_platform_to_ned = rpy_to_dcm(rpy0)
    C_sensor_to_ned = C_platform_to_ned @ C_INS_TO_PLATFORM
    inertial = Inertial(
        StandardPosVelAtt(t0, llh0, vned0, C_sensor_to_ned),
        MechanizationOptions(
            GravModels.SCHWARTZ,
            EarthModels.ELLIPTICAL,
            DcmIntegrationMethods.SIXTH_ORDER,
            IntegrationMethods.RECTANGULAR,
        ),
    )

    for k in range(len(imu_time)):
        t = imu_time[k]
        dv = dv_t[k]
        dth = dth_t[k]
        inertial.mechanize(t, C_INS_TO_PLATFORM @ dv, C_INS_TO_PLATFORM @ dth)
        sol = inertial.get_solution()
        sol_numpy = np.concatenate(
            [sol.get_llh(), sol.get_vned(), dcm_to_rpy(sol.get_C_s_to_ned())]
        )
        out[k + 1] = sol_numpy

    return out


def extract_pva_from_log(log):
    pva_time = []
    pva_t = []

    print('Reading PVA measurements from log...')

    for msg in log:
        if msg.channel == PVA_CHANNEL:
            t, aspn_msg = decode_aspn_lcm_msg(msg)
            llh, vned, rpy, _, _, _ = get_pva(aspn_msg)
            pva_time.append(t)
            pva_t.append(np.concatenate([llh, vned, rpy]))

    pva_time = np.array(pva_time)
    pva_t = np.array(pva_t)

    return pva_time, pva_t


def extract_imu_from_log(log):
    imu_time = []
    dv_t = []
    dth_t = []

    print('Reading IMU measurements from log...')

    for msg in log:
        if msg.channel == IMU_CHANNEL:
            t, aspn_msg = decode_aspn_lcm_msg(msg)
            dv, dth = get_imu(aspn_msg)
            imu_time.append(t)
            dv_t.append(dv)
            dth_t.append(dth)

    dv_t = np.array(dv_t)
    dth_t = np.array(dth_t)

    return imu_time, dv_t, dth_t


def generate_imu_from_pva(pva_time, pva):
    print('Generating IMU measurements from PVA...')
    forces, rates = inu.inv_mech(pva[:, :3], pva[:, 6:9], IMU_DT)

    imu_time = pva_time + IMU_DT
    imu_time = np.array([to_type_timestamp(t) for t in imu_time])

    return imu_time, forces.copy(), rates.copy()


def main(logfile: str) -> None:
    log = EventLog(logfile, 'r')

    # Extract PVA
    abs_truth_time, truth = extract_pva_from_log(log)
    t0 = abs_truth_time[0]
    truth_time = np.array([to_seconds(t - t0) for t in abs_truth_time])
    truth_dt = np.diff(truth_time)
    llh0 = truth[0, :3]
    rpy0 = truth[0, 6:9]
    truth_ned = llh_to_ned(truth[:, :3], llh0)

    interp_truth, interp_time = interpolate_pva_advanced(
        llh_t=np.hstack((truth_time[:, np.newaxis], truth[:, :3])),
        rpy_t=np.hstack((truth_time[:, np.newaxis], truth[:, 6:])),
        dt=IMU_DT,
        vel_t=np.hstack((truth_time[:, np.newaxis], truth[:, 3:6])),
        interp_type='spline',
    )
    # Calculate initial velocity from initial truth positions
    vned0 = inu.llh_to_vne(interp_truth[:4, :3], interp_time[1] - interp_time[0])[
        0
    ].copy()

    print('Initial PVA:')
    print(f'\tTime: {t0}')
    print(f'\tPos: {llh0}')
    print(f'\tVel: {vned0}')
    print(f'\tRpy: {rpy0}')

    # Extract IMU (from log or generated from PVA)
    if IMU_CHANNEL is None:
        # Generate IMU measurements with relative timestamps and add in t0
        # afterward for improved precision.
        imu_time, forces, rates = generate_imu_from_pva(interp_time, interp_truth)
        dv = forces * IMU_DT
        dth = rates * IMU_DT
        imu_time += t0

    else:
        imu_time, dv, dth = extract_imu_from_log(log)
        forces = dv / IMU_DT
        rates = dth / IMU_DT

    # Mechanize with navtk
    navtk_pva = navtk_mechanize(t0, imu_time, llh0, vned0, rpy0, dv, dth)
    navtk_ned = llh_to_ned(navtk_pva[:, :3], llh0)

    # Mechanize with INU library
    imu_time = np.array([to_seconds(t - t0) for t in imu_time])
    plt.figure('Accel')
    plt.suptitle('Accel')
    plt.gca().remove()
    plt.subplot(3, 1, 1)
    plt.plot(imu_time, dv[:, 0])
    plt.subplot(3, 1, 2)
    plt.plot(imu_time, dv[:, 1])
    plt.subplot(3, 1, 3)
    plt.plot(imu_time, dv[:, 2])
    plt.xlabel('Time (s)')
    plt.subplot(3, 1, 1)
    plt.ylabel('X (m/s)')
    plt.subplot(3, 1, 2)
    plt.ylabel('Y (m/s)')
    plt.subplot(3, 1, 3)
    plt.ylabel('Z (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.figure('Gyro')
    plt.suptitle('Gyro')
    plt.gca().remove()
    plt.subplot(3, 1, 1)
    plt.plot(imu_time, dth[:, 0])
    plt.subplot(3, 1, 2)
    plt.plot(imu_time, dth[:, 1])
    plt.subplot(3, 1, 3)
    plt.plot(imu_time, dth[:, 2])
    plt.xlabel('Time (s)')
    plt.subplot(3, 1, 1)
    plt.ylabel('X (rad)')
    plt.subplot(3, 1, 2)
    plt.ylabel('Y (rad)')
    plt.subplot(3, 1, 3)
    plt.ylabel('Z (rad)')
    plt.legend()
    plt.tight_layout()

    print('Mechanizing with INU...')
    llh_t, vne_t, rpy_t = inu.mech(forces, rates, llh0, vned0, rpy0, IMU_DT)
    afit_ned = llh_to_ned(llh_t, llh0)

    # Show results
    delta_pos_steps = np.linalg.norm(np.diff(truth_ned[:, :2], axis=0), axis=1)
    delta_pos = np.sum(delta_pos_steps)
    print(f'Distance Traveled: {delta_pos / 1000} km')

    plt.figure('Truth DT')
    plt.scatter(truth_time[1:], truth_dt)
    plt.xlabel('Time (s)')
    plt.ylabel('DT (s)')

    plt.figure('Northing vs Easting')
    plt.scatter(navtk_ned[:, 1], navtk_ned[:, 0], marker='^', label='IMU')
    plt.scatter(truth_ned[:, 1], truth_ned[:, 0], marker='o', label='Truth')
    plt.scatter(afit_ned[:, 1], afit_ned[:, 0], marker='X', label='AFIT')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title('Northing vs. Easting')
    # plt.colorbar(label='Time (s)')
    plt.legend()

    final_pos_err = truth_ned[-1] - navtk_ned[-2]
    print('Final Pos Error:', final_pos_err)
    print(
        f'Percent Distance Traveled Error: {np.linalg.norm(final_pos_err) / delta_pos * 100:.2f}'
    )

    plt.show()


if __name__ == '__main__':
    logfile = sys.argv[1]
    main(logfile)
