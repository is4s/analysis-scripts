#!/usr/bin/env python3

import argparse
import os
from math import cos, radians, sin

import matplotlib.pyplot as plt
import numpy as np
from analysis.lcm.data import LogData, RangeRateData, VelData
from analysis.lcm.interpolation import interpolate_array
from analysis.lcm.log_readers import read_range_rate_to_point
from analysis.lcm.logfiles import sort_log
from analysis.lcm.plots import Plot
from aspn23_lcm import measurement_velocity
from lcm import Event, EventLog
from matplotlib.animation import FuncAnimation


def create_animation(data: RangeRateData):
    print('Running animation. Press any key to pause.')
    num_frames = len(data.time)

    fig = plt.figure('Point Cloud')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Right (m)')
    ax.set_xlim(-50, 50)
    ax.set_ylabel('Forward (m)')
    ax.set_ylim(0, 200)
    ax.set_zlabel('Up (m)')
    ax.set_zlim(-10, 10)
    cloud = ax.scatter([], [], [], s=10, c=[], cmap='plasma', vmin=0, vmax=30)
    plt.colorbar(cloud, ax=ax, label='Range Rate (m/s)')
    text = fig.text(
        0,
        1,
        f'Time: {data.time[0]:.2f} of {data.time[-1]:.2f}',
        transform=ax.transAxes,
    )

    def init():
        cloud._offsets3d = ([], [], [])
        cloud.set_array([])
        return (cloud, text)

    def update(frame):
        points = np.array(data.points[frame])
        x, y, z = points[:, 1], points[:, 0], -points[:, 2]
        cloud._offsets3d = (x, y, z)
        range_rates = -data.range_rates[frame]
        cloud.set_array(range_rates)
        text.set_text(f'Time: {data.time[frame]:.2f}s of {data.time[-1]:.2f}s')
        return (cloud, text)

    ani = FuncAnimation(
        fig, update, frames=num_frames, init_func=init, interval=5, repeat=False
    )
    paused = False

    def toggle_pause(*args, **kwargs):
        nonlocal paused

        if ani.event_source is None:
            return  # Plot closed

        if paused:
            ani.resume()
        else:
            ani.pause()
        paused = not paused

    fig.canvas.mpl_connect('key_press_event', toggle_pause)


def solve_least_squares(
    z: np.ndarray,
    H: np.ndarray,
    z_covariance: np.ndarray,
    verify_invertible: bool = True,
    reject_outliers: bool = True,
    max_condition_number_to_invert: float = 1.0e4,
    outlier_rejection_num_sigma: float = 3.0,
):
    # Initialize return variables
    solution = None
    solution_covariance = None
    residuals = None
    num_states = H.shape[1]

    done = False
    i_good = list(range(len(z)))

    # Don't even try if don't have enough measurements
    if reject_outliers:
        if len(i_good) < num_states + 1:
            done = True
    else:
        if len(i_good) < num_states:
            done = True

    while not done:
        this_z = z[i_good]
        this_H = H[i_good, :]
        R = z_covariance[i_good, :][:, i_good]

        W = np.linalg.inv(R)
        M_inv = this_H.T @ W @ this_H
        if verify_invertible:
            done = np.linalg.cond(M_inv) > max_condition_number_to_invert

        if not done:
            M = np.linalg.inv(this_H.T @ W @ this_H)
            x = M @ this_H.T @ W @ this_z

            if reject_outliers:
                resids = this_z - this_H @ x
                idempotent_matrix = this_H @ M @ this_H.T @ W
                resid_covariance = (np.eye(len(i_good)) - idempotent_matrix) @ R
                resid_sigma = np.sqrt(np.diag(resid_covariance))
                resid_sigma = resids / resid_sigma
                if max(resid_sigma) <= outlier_rejection_num_sigma:
                    solution = x
                    solution_covariance = M
                    residuals = resids
                    done = True
                else:
                    # remove the one with the highest residual relative to sigma
                    highest_index = np.argmax(resid_sigma)
                    i_to_reject = i_good[highest_index]
                    i_good.remove(i_to_reject)
                    if len(i_good) < num_states + 1:
                        done = True  # Not enough measurements to check residuals
            else:
                solution = x
                solution_covariance = M
                residuals = this_z - this_H @ x
                done = True

    return solution, solution_covariance, residuals


def plot_vel(data: VelData, truth: VelData):
    speed = np.linalg.norm(data.vel, axis=1)
    truth_speed = np.linalg.norm(truth.vel, axis=1)

    speed_plot = Plot(
        title=f'{data.label} Speed',
        xlabel='Time (s)',
        ylabels='Speed (m/s)',
    )
    speed_plot.add_data(
        truth.label, truth.time, truth_speed, is_scatter=True, marker='.'
    )
    speed_plot.add_data(data.label, data.time, speed, is_scatter=True, marker='.')
    speed_plot.plot()

    vel_plot = Plot(
        title=f'{data.label} Velocity',
        xlabel='Time (s)',
        ylabels=['Forward (m/s)', 'Lateral (m/s)', 'Vertical (m/s)'],
    )
    vel_plot.add_data(truth.label, truth.time, truth.vel, is_scatter=True, marker='.')
    vel_plot.add_data(data.label, data.time, data.vel, is_scatter=True, marker='.')
    vel_plot.plot()

    truth_vel = interpolate_array(truth.time, truth.vel, data.time)
    truth_speed = np.linalg.norm(truth_vel, axis=1)
    speed_err = data.vel - truth_vel
    speed_err_plot = Plot(
        title=f'{data.label} Speed Error',
        xlabel='Time (s)',
        ylabels='Speed Error (m/s)',
    )
    speed_err_plot.add_data(
        data.label, data.time, speed_err, is_scatter=True, marker='.'
    )
    speed_err_plot.plot()

    vel_err = data.vel - truth_vel
    vel_err_plot = Plot(
        title=f'{data.label} Velocity Error',
        xlabel='Time (s)',
        ylabels=['Forward (m/s)', 'Lateral (m/s)', 'Vertical (m/s)'],
    )
    vel_err_plot.add_data(truth.label, data.time, vel_err, is_scatter=True, marker='.')
    vel_err_plot.plot()
    vel_err_w_sigma_plot = Plot(
        title=f'{data.label} Velocity Error w/ Sigma',
        xlabel='Time (s)',
        ylabels=['Forward (m/s)', 'Lateral (m/s)', 'Vertical (m/s)'],
        legend=[data.label, '+/- 1 sigma'],
    )
    vel_err_w_sigma_plot.add_data(
        truth.label, data.time, vel_err, is_scatter=True, marker='.'
    )
    vel_err_w_sigma_plot.add_data(
        data.label, data.time, data.sig, color='k', linestyle='--'
    )
    vel_err_w_sigma_plot.add_data(
        data.label, data.time, -data.sig, color='k', linestyle='--'
    )
    vel_err_w_sigma_plot.plot()


def plot_info(radar_time, vel_time, num_targets, resid_mat):
    plt.figure('Number of targets')
    plt.suptitle('Number of targets')
    plt.plot(radar_time, num_targets, '.')

    plt.figure('Residuals')
    plt.suptitle('Residuals')
    plt.plot(vel_time, resid_mat, '.')


def calc_vel(data: RangeRateData) -> VelData:
    num_targets = np.array([len(rates) for rates in data.range_rates])
    max_num_targets = max(num_targets)

    x_vel = np.zeros((len(data.time), max_num_targets))
    x_vel.fill(np.nan)

    # forward vel in sensor frame
    az = radians(0.0)
    elev = radians(0.0)  # radians(61.0)
    p_forward = np.array([cos(elev) * cos(az), cos(elev) * sin(az), sin(elev)])
    p_forward = p_forward / np.linalg.norm(p_forward)  # make a unit vector

    vel_data = VelData('Radar')
    vel_data.vel = np.full((len(data.time), 3), np.nan)
    vel_data.sig = np.full((len(data.time), 3), np.nan)

    resid_mat = np.full((len(data.time), max_num_targets), np.nan)

    count = 0

    for j in range(len(data.time)):
        # Calculate normalized "forward" velocity for each target
        for target_num in range(num_targets[j]):
            target_coords = np.array(data.points[j][target_num])
            pointing_vec = target_coords / np.linalg.norm(target_coords)
            temp = p_forward * pointing_vec
            x_vel[j, target_num] = -data.range_rates[j][target_num] / temp[0]

        # Only use range rate measurements that reasonably match forward velocities
        median_value = np.median(x_vel[j, : len(data.range_rates[j])])
        i_good = np.where(
            np.abs(x_vel[j, : len(data.range_rates[j])] - median_value) < 1.3
        )[0]

        if len(i_good) >= 4:
            # Measurement vector
            z = -np.array(np.array(data.range_rates[j])[i_good])

            # Convert points to unit vectors (H matrix for least squares)
            H = np.array(data.points[j])[i_good]
            for k in range(H.shape[0]):
                H[k, :] = H[k, :] / np.linalg.norm(H[k, :])

            # Measurement covariance matrix
            z_covariance = np.diag(0.05**2 * np.ones(len(z)))

            # Optional: Could use this instead if want more advanced features (resitual rejection)
            # Current settings are equivalient to unweighted above
            use_least_squares_function = False
            if use_least_squares_function:
                vel, cov, res = solve_least_squares(
                    z=z,
                    H=H,
                    z_covariance=z_covariance,
                    verify_invertible=False,
                    reject_outliers=False,
                )
            else:
                # Standard unweighted least squares solution
                cov = np.linalg.inv(H.T @ H)
                vel = cov @ H.T @ z
                res = z - H @ vel

            if vel is not None:
                vel_data.vel[j, :3] = vel
                vel_data.sig[j, :] = np.sqrt(np.diag(cov))
                resid_mat[j, : len(res)] = res
                count += 1

    nonnan_indices = ~np.any(np.isnan(vel_data.vel), axis=1)
    vel_data.time = data.time[nonnan_indices]
    vel_data.vel = vel_data.vel[nonnan_indices]
    vel_data.sig = vel_data.sig[nonnan_indices]
    resid_mat = resid_mat[nonnan_indices]

    plot_info(data.time, vel_data.time, num_targets, resid_mat)

    return vel_data


def save_vel(logfile: str, t0: int, vel_data: VelData):
    print('Saving velocity measurements to new log...')

    basename, ext = os.path.splitext(logfile)
    log = EventLog(logfile)
    out_filename = f'{basename}_with_vel{ext}'
    out_log = EventLog(out_filename, 'w', True)

    msg: Event
    for msg in log:
        out_log.write_event(msg.timestamp, msg.channel, msg.data)
    for time, vel, sig in zip(vel_data.time, vel_data.vel, vel_data.sig):
        lcm_vel = measurement_velocity()
        lcm_vel.time_of_validity.elapsed_nsec = round(time * 1_000_000_000 + t0)
        lcm_vel.num_meas = 3
        lcm_vel.x = vel[0]
        lcm_vel.y = vel[1]
        lcm_vel.z = vel[2]
        lcm_vel.covariance = np.diag(np.square(sig))
        lcm_vel.reference_frame = lcm_vel.REFERENCE_FRAME_SENSOR
        lcm_vel.num_error_model_params = 0
        lcm_vel.error_model_params = []
        lcm_vel.num_integrity = 0
        lcm_vel.integrity = []
        lcm_time = int(lcm_vel.time_of_validity.elapsed_nsec * 1e-3)
        out_log.write_event(lcm_time, '/sensor/drvegrd171/velocity', lcm_vel.encode())
    log.close()
    out_log.close()

    sort_log(out_filename)


def plot_range_rate_to_point(log_data: LogData[RangeRateData | VelData]) -> None:
    truth: VelData = log_data.data[log_data.truth_channel]

    for channel, data in log_data.data.items():
        if channel == log_data.truth_channel:
            continue

        vel_data: VelData = calc_vel(data)
        save_vel(log_data.logfile, log_data.t0.get_elapsed_nsec(), vel_data)
        plot_vel(vel_data, truth)
        # Filter out points with large covariance and replot
        filtered_indices = np.where(np.all(vel_data.sig < 5, axis=1))
        vel_data.label = 'Filtered Radar'
        vel_data.time = vel_data.time[filtered_indices]
        vel_data.vel = vel_data.vel[filtered_indices]
        vel_data.sig = vel_data.sig[filtered_indices]
        plot_vel(vel_data, truth)

        create_animation(data)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="""Plot range rate messages from LCM log."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all range rate messages in log. If not set, will prompt user to determine which range rate channels should be plotted.',
        action='store_true',
    )

    args = parser.parse_args()
    log_data = read_range_rate_to_point(args.logfile, args.all)
    plot_range_rate_to_point(log_data)


if __name__ == '__main__':
    main()
