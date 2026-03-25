import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from analysis.lcm.data import PvaData
from analysis.lcm.error import calc_drms, calc_tilts
from analysis.lcm.interpolation import interpolate_pva

from .utils import show_stats


def plot_pva(
    pva: PvaData, truth_pva: PvaData, t0: float, save_dir: str | None = None
) -> None:
    """Generate position, velocity and attitude plots, as well as error plots.

    Args:
        - pva: Main PVA to plot
        - truth_pva: Reference PVA to plot. Also used to calculate and plot errors of :param:`pva`
        - save_dir: Directory to save plots to, if desired. If None, will not save plots.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    leg = [pva.label, truth_pva.label]
    leg_w_sigma = [pva.label, truth_pva.label, '+/- 1 sigma']

    # Interpolate truth onto solution times so that we can calulate the solution error
    interp_truth_pva = interpolate_pva(pva, truth_pva)
    ned_err = pva.ned - interp_truth_pva.ned
    vel_err = pva.vel - interp_truth_pva.vel
    rpy_rad = np.deg2rad(pva.rpy)
    truth_rpy_rad = np.deg2rad(interp_truth_pva.rpy)
    tilt_err = np.rad2deg(calc_tilts(truth_rpy_rad, rpy_rad))

    drms = calc_drms(ned_err[:, 0], ned_err[:, 1])
    drms_str = f'DRMS: {drms:.2f} m'

    plot_trajectory(
        pva.ned, truth_pva.ned, pva.time, truth_pva.time, leg, drms_str, save_dir
    )

    plot_llh(
        pva.llh,
        truth_pva.llh,
        pva.llh_sig,
        pva.time,
        truth_pva.time,
        t0,
        leg_w_sigma,
        save_dir,
    )
    plot_ned(
        pva.ned,
        truth_pva.ned,
        pva.ned_sig,
        pva.time,
        truth_pva.time,
        t0,
        leg_w_sigma,
        save_dir,
    )
    plot_ned_err(ned_err, pva.ned_sig, pva.time, t0, pva.label, save_dir)

    plot_vel(
        pva.vel,
        truth_pva.vel,
        pva.vel_sig,
        pva.time,
        truth_pva.time,
        t0,
        leg_w_sigma,
        save_dir,
    )
    plot_vel_err(vel_err, pva.vel_sig, pva.time, t0, pva.label, save_dir)

    plot_rpy(
        pva.rpy,
        truth_pva.rpy,
        pva.tilt_sig,
        pva.time,
        truth_pva.time,
        t0,
        leg_w_sigma,
        save_dir,
    )
    plot_tilt_err(tilt_err, pva.tilt_sig, pva.time, t0, pva.label, save_dir)


def plot_trajectory(
    ned: NDArray,
    truth_ned: NDArray,
    time: NDArray,
    truth_time: NDArray,
    leg: list[str],
    drms_str: str,
    plot_dir: str | None = None,
):
    fig = plt.figure('Trajectory')
    plt.suptitle('Northing vs. Easting')
    # Get subset of Blues and Oranges colormaps, so that the last points aren't
    # too light-colored to be seen.
    blues = plt.get_cmap('Blues_r')
    cmap_colors = blues(np.linspace(0.0, 0.80, 256))
    blues = LinearSegmentedColormap.from_list('NewBlues_r', cmap_colors)
    oranges = plt.get_cmap('Oranges_r')
    cmap_colors = oranges(np.linspace(0.0, 0.80, 256))
    oranges = LinearSegmentedColormap.from_list('NewOranges_r', cmap_colors)
    plt.scatter(ned[:, 1], ned[:, 0], marker='o', cmap=blues, c=time)
    plt.scatter(
        truth_ned[:, 1], truth_ned[:, 0], marker='o', cmap=oranges, c=truth_time
    )
    plt.axis('equal')
    plt.ylabel('Northing (m)')
    plt.xlabel('Easting (m)')
    plt.legend(leg)
    cb = plt.colorbar()
    cb.set_label('Time (s)')
    # Shrink colorbar vertically so we can insert DRMS above it
    cb.ax.set_position(
        [
            cb.ax.get_position().x0,
            cb.ax.get_position().y0,
            cb.ax.get_position().width,
            cb.ax.get_position().height * 0.9,
        ]
    )
    # Add error metrics above the colorbar
    fig.text(
        cb.ax.get_position().x0 + cb.ax.get_position().width / 2 + 0.02,
        cb.ax.get_position().y0 + cb.ax.get_position().height + 0.02,
        drms_str,
        ha='center',
        va='bottom',
        fontsize=10,
    )
    if plot_dir:
        filename = os.path.join(plot_dir, 'ne_trajectory')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_llh(
    llh: NDArray,
    truth_llh: NDArray,
    llh_sig: NDArray,
    time: NDArray,
    truth_time: NDArray,
    t0: float,
    leg: list[str],
    plot_dir: str | None = None,
):
    plt.figure('LLA Pos')
    plt.suptitle('LLA Position and Uncertainty vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, llh[:, 0])
    plt.plot(truth_time, truth_llh[:, 0], 'g')
    plt.plot(time, llh[:, 0] + llh_sig[:, 0], 'black', linestyle='--')
    plt.plot(time, llh[:, 0] - llh_sig[:, 0], 'black', linestyle='--')
    plt.ylabel('Latitude (rad)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, llh[:, 1])
    plt.plot(truth_time, truth_llh[:, 1], 'g')
    plt.plot(time, llh[:, 1] + llh_sig[:, 1], 'black', linestyle='--')
    plt.plot(time, llh[:, 1] - llh_sig[:, 1], 'black', linestyle='--')
    plt.ylabel('Longitude (rad)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, llh[:, 2])
    plt.plot(truth_time, truth_llh[:, 2], 'g')
    plt.plot(time, llh[:, 2] + llh_sig[:, 2], 'black', linestyle='--')
    plt.plot(time, llh[:, 2] - llh_sig[:, 2], 'black', linestyle='--')
    plt.ylabel('Altitude (m)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(leg)

    if plot_dir:
        filename = os.path.join(plot_dir, 'lla_position')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_ned(
    ned: NDArray,
    truth_ned: NDArray,
    ned_sig: NDArray,
    time: NDArray,
    truth_time: NDArray,
    t0: float,
    leg: list[str],
    plot_dir: str | None = None,
):
    plt.figure('NED Pos')
    plt.suptitle('NED Position vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, ned[:, 0])
    plt.plot(truth_time, truth_ned[:, 0], 'g')
    plt.plot(time, ned[:, 0] + ned_sig[:, 0], 'black', linestyle='--')
    plt.plot(time, ned[:, 0] - ned_sig[:, 0], 'black', linestyle='--')
    plt.ylabel('Northing (m)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, ned[:, 1])
    plt.plot(truth_time, truth_ned[:, 1], 'g')
    plt.plot(time, ned[:, 1] + ned_sig[:, 1], 'black', linestyle='--')
    plt.plot(time, ned[:, 1] - ned_sig[:, 1], 'black', linestyle='--')
    plt.ylabel('Easting (m)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, ned[:, 2])
    plt.plot(truth_time, truth_ned[:, 2], 'g')
    plt.plot(time, ned[:, 2] + ned_sig[:, 2], 'black', linestyle='--')
    plt.plot(time, ned[:, 2] - ned_sig[:, 2], 'black', linestyle='--')
    plt.ylabel('Down (m)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(leg)
    if plot_dir:
        filename = os.path.join(plot_dir, 'ned_position')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_ned_err(
    ned_err: NDArray,
    ned_sig: NDArray,
    time: NDArray,
    t0: float,
    solution_label: str,
    plot_dir: str | None = None,
):
    fig = plt.figure('NED Pos Error')
    plt.suptitle(f'{solution_label} NED Position Error vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, ned_err[:, 0])
    plt.plot(time, ned_sig[:, 0], 'black', linestyle='--')
    plt.plot(time, -ned_sig[:, 0], 'black', linestyle='--')
    show_stats(fig, ned_err[:, 0])
    plt.ylabel('North Error (m)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, ned_err[:, 1])
    plt.plot(time, ned_sig[:, 1], 'black', linestyle='--')
    plt.plot(time, -ned_sig[:, 1], 'black', linestyle='--')
    show_stats(fig, ned_err[:, 1])
    plt.ylabel('East Error (m)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, ned_err[:, 2])
    plt.plot(time, ned_sig[:, 2], 'black', linestyle='--')
    plt.plot(time, -ned_sig[:, 2], 'black', linestyle='--')
    show_stats(fig, ned_err[:, 2])
    plt.ylabel('Down Error (m)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(['Error', '+/- 1-sigma'])
    if plot_dir:
        filename = os.path.join(plot_dir, 'ned_position_error')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_vel(
    vel: NDArray,
    truth_vel: NDArray,
    vel_sig: NDArray,
    time: NDArray,
    truth_time: NDArray,
    t0: float,
    leg: list[str],
    plot_dir: str | None = None,
):
    plt.figure('Vel')
    plt.suptitle('Velocity and Uncertainty vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, vel[:, 0])
    plt.plot(truth_time, truth_vel[:, 0], 'g')
    plt.plot(time, vel[:, 0] + vel_sig[:, 0], 'black', linestyle='--')
    plt.plot(time, vel[:, 0] - vel_sig[:, 0], 'black', linestyle='--')
    plt.ylabel('North vel (m/s)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, vel[:, 1])
    plt.plot(truth_time, truth_vel[:, 1], 'g')
    plt.plot(time, vel[:, 1] + vel_sig[:, 1], 'black', linestyle='--')
    plt.plot(time, vel[:, 1] - vel_sig[:, 1], 'black', linestyle='--')
    plt.ylabel('East vel (m/s)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, vel[:, 2])
    plt.plot(truth_time, truth_vel[:, 2], 'g')
    plt.plot(time, vel[:, 2] + vel_sig[:, 2], 'black', linestyle='--')
    plt.plot(time, vel[:, 2] - vel_sig[:, 2], 'black', linestyle='--')
    plt.ylabel('Down vel (m/s)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(leg)
    if plot_dir:
        filename = os.path.join(plot_dir, 'ned_velocity')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_vel_err(
    vel_err: NDArray,
    vel_sig: NDArray,
    time: NDArray,
    t0: float,
    solution_label: str,
    plot_dir: str | None = None,
):
    fig = plt.figure('NED Vel Error')
    plt.suptitle(f'{solution_label} NED Velocity Error vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, vel_err[:, 0])
    plt.plot(time, vel_sig[:, 0], 'black', linestyle='--')
    plt.plot(time, -vel_sig[:, 0], 'black', linestyle='--')
    show_stats(fig, vel_err[:, 0])
    plt.ylabel('North Error (m/s)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, vel_err[:, 1])
    plt.plot(time, vel_sig[:, 1], 'black', linestyle='--')
    plt.plot(time, -vel_sig[:, 1], 'black', linestyle='--')
    show_stats(fig, vel_err[:, 1])
    plt.ylabel('East Error (m/s)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, vel_err[:, 2])
    plt.plot(time, vel_sig[:, 2], 'black', linestyle='--')
    plt.plot(time, -vel_sig[:, 2], 'black', linestyle='--')
    show_stats(fig, vel_err[:, 2])
    plt.ylabel('Down Error (m/s)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(['Error', '+/- 1-sigma'])
    if plot_dir:
        filename = os.path.join(plot_dir, 'ned_velocity_error')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_rpy(
    rpy: NDArray,
    truth_rpy: NDArray,
    tilt_sig: NDArray,
    time: NDArray,
    truth_time: NDArray,
    t0: float,
    leg: list[str],
    plot_dir: str | None = None,
):
    plt.figure('RPY')
    plt.suptitle('RPY and (tilt) Uncertainty vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, rpy[:, 0])
    plt.plot(truth_time, truth_rpy[:, 0], 'g')
    plt.plot(time, (rpy[:, 0] + tilt_sig[:, 0]), 'black', linestyle='--')
    plt.plot(time, (rpy[:, 0] - tilt_sig[:, 0]), 'black', linestyle='--')
    plt.ylabel('Roll (deg)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, rpy[:, 1])
    plt.plot(truth_time, truth_rpy[:, 1], 'g')
    plt.plot(time, rpy[:, 1] + (tilt_sig[:, 1]), 'black', linestyle='--')
    plt.plot(time, rpy[:, 1] - (tilt_sig[:, 1]), 'black', linestyle='--')
    plt.ylabel('Pitch (deg)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, rpy[:, 2])
    plt.plot(truth_time, truth_rpy[:, 2], 'g')
    plt.plot(time, rpy[:, 2] + (tilt_sig[:, 2]), 'black', linestyle='--')
    plt.plot(time, rpy[:, 2] - (tilt_sig[:, 2]), 'black', linestyle='--')
    plt.ylabel('Yaw (deg)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(leg)
    if plot_dir:
        filename = os.path.join(plot_dir, 'rpy')
        plt.savefig(f'{filename}.png', dpi=300)


def plot_tilt_err(
    tilts: NDArray,
    tilt_sig: NDArray,
    time: NDArray,
    t0: float,
    solution_label: str,
    plot_dir: str | None = None,
):
    fig = plt.figure('NED Tilt Error')
    plt.suptitle(f'{solution_label} NED Tilt Error vs. Time')
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time, tilts[:, 0])
    plt.plot(time, tilt_sig[:, 0], 'black', linestyle='--')
    plt.plot(time, -tilt_sig[:, 0], 'black', linestyle='--')
    show_stats(fig, tilts[:, 0])
    plt.ylabel('North tilt (deg)')
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time, tilts[:, 1])
    plt.plot(time, tilt_sig[:, 1], 'black', linestyle='--')
    plt.plot(time, -tilt_sig[:, 1], 'black', linestyle='--')
    show_stats(fig, tilts[:, 1])
    plt.ylabel('East tilt (deg)')
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(time, tilts[:, 2])
    plt.plot(time, tilt_sig[:, 2], 'black', linestyle='--')
    plt.plot(time, -tilt_sig[:, 2], 'black', linestyle='--')
    show_stats(fig, tilts[:, 2])
    plt.ylabel('Down tilt (deg)')
    ax1.sharex(ax2)
    ax2.sharex(ax3)
    plt.xlabel(f'Relative time (sec), t0 = {t0}')
    plt.legend(['Error', '+/- 1-sigma'])
    if plot_dir:
        filename = os.path.join(plot_dir, 'ned_tilt_error')
        plt.savefig(f'{filename}.png', dpi=300)
