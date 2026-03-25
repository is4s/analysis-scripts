#!/usr/bin/env python3

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tomlkit
from analysis.lcm.config import CONFIG_FILE
from analysis.lcm.data import MagData
from analysis.lcm.log_formats import ERROR, INFO
from analysis.lcm.log_readers import MagLogReader
from aspn23_lcm import measurement_magnetic_field
from navtk.magnetic import (
    MagnetometerCalibrationCaruso2d,
    MagnetometerCalibrationEllipse2d,
    mag_to_heading,
)


def plot_mag_data(data: MagData, prefix: str):
    # Plot XY mag field
    plt.figure(f'{prefix} XY Magnetic Field')
    plt.plot(data.mag[:, 0], data.mag[:, 1], label=data.label)

    # Plot mag field vs time
    plt.figure(f'{prefix} Magnetic Field')
    plt.subplot(3, 1, 1)
    plt.plot(data.time, data.mag[:, 0], label=data.label)
    plt.subplot(3, 1, 2)
    plt.plot(data.time, data.mag[:, 1], label=data.label)
    plt.subplot(3, 1, 3)
    plt.plot(data.time, data.mag[:, 2], label=data.label)


def plot_heading(data: MagData, truth_data: MagData):
    # Plot heading vs time
    plt.figure('Heading')
    plt.plot(data.time, data.heading, label=data.label)

    # Plot heading error vs time if truth data is available, and this data is not truth
    if len(truth_data.time) == 0 or data.label == truth_data.label:
        return

    truth_heading = np.unwrap(truth_data.heading, period=360)
    unwrapped_heading = np.unwrap(data.heading, period=360)
    interp_heading = np.interp(truth_data.time, data.time, unwrapped_heading)
    heading_error = (((interp_heading - truth_heading) + 180) % 360) - 180
    plt.figure('Heading Error')
    plt.title('Heading Error')
    plt.plot(truth_data.time, heading_error, label=data.label)

    plt.figure('Heading Error vs Reported Heading')
    plt.title('Heading Error vs Reported Heading')
    plt.scatter(interp_heading % 360, heading_error, label=data.label, marker='.')

    plt.figure('Heading Error vs True Heading')
    plt.title('Heading Error vs True Heading')
    plt.scatter(truth_heading % 360, heading_error, label=data.label, marker='.')


def setup_plots():
    for prefix in ['Raw', 'Rotated', 'Calibrated']:
        plt.figure(f'{prefix} XY Magnetic Field')
        plt.title(f'{prefix} XY Magnetic Field')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.figure(f'{prefix} Magnetic Field')
        plt.title(f'{prefix} Magnetic Field')
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.ylabel('X')
        plt.subplot(3, 1, 2)
        plt.ylabel('Y')
        plt.subplot(3, 1, 3)
        plt.ylabel('Z')

    plt.figure('Heading')
    plt.title('Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (deg)')

    plt.figure('Heading Error')
    plt.title('Heading Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Error (deg)')

    plt.figure('Heading Error vs Reported Heading')
    plt.title('Heading Error vs Reported Heading')
    plt.xlabel('Heading (deg)')
    plt.ylabel('Heading Error (deg)')

    plt.figure('Heading Error vs True Heading')
    plt.title('Heading Error vs True Heading')
    plt.xlabel('Heading (deg)')
    plt.ylabel('Heading Error (deg)')


def finish_plots():
    for prefix in ['Raw', 'Rotated', 'Calibrated']:
        plt.figure(f'{prefix} XY Magnetic Field')
        plt.legend()
        plt.tight_layout()
        plt.axis('equal')

        plt.figure(f'{prefix} Magnetic Field')
        plt.legend()
        plt.tight_layout()

    plt.figure('Heading')
    plt.legend()
    plt.tight_layout()

    if plt.fignum_exists('Heading Error'):
        plt.figure('Heading Error')
        plt.legend()
        plt.tight_layout()

        plt.figure('Heading Error vs Reported Heading')
        plt.legend()
        plt.tight_layout()

        plt.figure('Heading Error vs True Heading')
        plt.legend()
        plt.tight_layout()


def apply_rotation(data: MagData, config: dict[str, Any]):
    channel = data.label
    if channel not in config:
        print(f'{ERROR}: No mag config specified for channel {channel}. Terminating.')
        exit(1)
    if 'sensor_to_platform' not in config[channel]:
        print(
            f'{ERROR}: No mag to platform rotation specified for channel {channel}. Terminating.'
        )
        exit(1)
    C_mag_to_platform = np.array(config[channel]['sensor_to_platform'])
    data.mag = (C_mag_to_platform @ data.mag.T).T


def apply_calibration(data: MagData, config: dict[str, Any], calibrate: bool):
    channel = data.label
    if channel not in config:
        print(f'{ERROR}: No mag config specified for channel {channel}. Terminating.')
        exit(1)
    if 'method' not in config[channel]:
        print(
            f'{ERROR}: No calibration method specified for channel {channel}. Terminating.'
        )
        exit(1)

    cal_method = config[channel]['method']
    match cal_method:
        case 'caruso':
            mag_calibration = MagnetometerCalibrationCaruso2d()
        case 'ellipse':
            mag_calibration = MagnetometerCalibrationEllipse2d()
        case 'both':
            mag_calibration = MagnetometerCalibrationEllipse2d(True)
        case _:
            print(
                f'{ERROR}: No mag calibration method specified for channel {channel}. Terminating.'
            )
            exit(1)

    if calibrate:
        print(
            f'{INFO} Generating calibration params for {channel} using {type(mag_calibration)}.'
        )
        mag_calibration.generate_calibration(data.mag.T)
        calibration_params = mag_calibration.get_calibration_params()
        data.scale_factor = calibration_params[0].tolist()
        data.bias = calibration_params[1].tolist()
    else:
        print(f'{INFO} Grabbing calibration params for {channel} from config.')
        calibration_params = config[channel]
        data.bias = calibration_params['bias']
        data.scale_factor = calibration_params['scale_factor']
        mag_calibration.set_calibration_params(data.scale_factor, data.bias)

    for idx in range(len(data.mag)):
        # Apply calibration to each measurement
        mag_meas = data.mag[idx]
        mag_meas = mag_calibration.apply_calibration(mag_meas)

        # Save newly calibrated data
        data.mag[idx] = mag_meas


def save_calibration(data: MagData, config: dict[str, Any], config_path: str):
    channel = data.label
    print(
        f'{INFO} Calibration params for channel {channel}:\n'
        f'\tScale Factor: {data.scale_factor}\n'
        f'\tBias: {data.bias}'
    )
    update_config = input('Update config file with these calibration params? [y/n]')
    if update_config == 'y':
        config[channel]['scale_factor'] = data.scale_factor
        config[channel]['bias'] = data.bias

        with open(config_path, 'w') as f:
            tomlkit.dump(config, f)

        print(f'{INFO} Updated params written to {config_path}')


def calc_mag_heading(data: MagData, config: dict):
    declination = config['mag_declination']
    heading = [mag_to_heading(x, y, declination) for x, y in data.mag[:, :2]]
    data.heading = np.rad2deg(heading)


def plot_mag(logfile: str, extract_all: bool, calibrate: bool) -> None:
    log_reader = MagLogReader(
        logfile,
        (
            measurement_magnetic_field,
        ),
        extract_all,
        CONFIG_FILE,
    )
    log_data = log_reader.read_log()
    mag_data = log_data.data
    truth_data = mag_data[log_reader.truth_channel]

    setup_plots()

    for data in mag_data.values():
        if data.mag.size:
            plot_mag_data(data, 'Raw')

            apply_rotation(data, log_reader.config)
            plot_mag_data(data, 'Rotated')

            apply_calibration(data, log_reader.config, calibrate)
            plot_mag_data(data, 'Calibrated')

            calc_mag_heading(data, log_reader.config)

        plot_heading(data, truth_data)

    finish_plots()

    plt.show()

    for data in mag_data.values():
        if calibrate and data.scale_factor is not None:
            save_calibration(data, log_reader.config, config_path)


def main():
    parser = argparse.ArgumentParser(
        description="""Plot magnetic heading measurements from LCM log."""
    )

    parser.add_argument('logfile', help='LCM log file.')
    parser.add_argument(
        '-a',
        '--all',
        help='Plot all magnetic field and heading messages in log. If not set, will prompt user to determine which magnetic field channels should be calibrated and plotted.',
        action='store_true',
    )
    parser.add_argument(
        '-c',
        '--calibrate',
        help='Calibrate magnetic field measurements before plotting. If not set, will try to use calibration parameters from config file.',
        action='store_true',
    )

    args = parser.parse_args()
    plot_mag(args.logfile, args.all, args.calibrate)


if __name__ == '__main__':
    main()
