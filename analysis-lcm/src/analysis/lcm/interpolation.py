import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import minimize_scalar

from .data import PvaData


def interpolate_array(x: np.ndarray, y: np.ndarray, x_interp: np.ndarray):
    return interp1d(x, y, bounds_error=False, fill_value=np.nan, axis=0, kind='cubic')(
        x_interp
    )


def interpolate_pva(pva1: PvaData, pva2: PvaData) -> PvaData:
    """Return the result of interpolating pva2 onto pva1 times."""
    out = PvaData(pva2.label)
    out.time = pva1.time
    out.llh = interpolate_array(pva2.time, pva2.llh, pva1.time)
    out.llh_sig = interpolate_array(pva2.time, pva2.llh_sig, pva1.time)
    out.ned = interpolate_array(pva2.time, pva2.ned, pva1.time)
    out.ned_sig = interpolate_array(pva2.time, pva2.ned_sig, pva1.time)
    out.vel = interpolate_array(pva2.time, pva2.vel, pva1.time)
    out.vel_sig = interpolate_array(pva2.time, pva2.vel_sig, pva1.time)
    out.rpy = interpolate_array(pva2.time, pva2.rpy, pva1.time)
    out.tilt_sig = interpolate_array(pva2.time, pva2.tilt_sig, pva1.time)

    return out


def interpolate_pva_advanced(
    llh_t: np.ndarray,
    rpy_t: np.ndarray,
    dt: float,
    vel_t: np.ndarray | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    interp_type: str | None = None,
    s: list[float | None] | None = None,
    w: list[float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate over PVA data to the specified data rate between the start and stop times.

    Parameters
    ----------
    llh_t
        timestamped position in a Nx4 array of the structure Time (s), Latitude (rad), Longitude (rad), HAE (m)
    vel_t
        timestamped velocity in a Nx4 array of the structure Time (s), Velocity north (m/s), Velocity east (m/s), Velocity down (m/s)
    rpy_t
        timestamped attitude in a Nx4 array of the structure Time (s), Roll (rad), Pitch (rad), Yaw (rad)
    dt
        time delta of desired interpolated numpy array
    t_start
        time of first point of interpolation
    t_stop
        time of last point of interpolation
    interp_type
        Type of interpolation to perform. Either zero, linear, slinear, quadratic, cubic, or spline.
        'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order
        see the `kind` argument in scipy interp1d https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    s
        list of smoothing values for each of the 9 fields of the PVA. Used for splines only.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html for more information.
    w
        list of weights for each of the 9 fields of the PVA. weights are used in computing the weighted least-squares spline fit.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html for more information.
    plot
        if True, plots comparing the input pva and interpolated pva are generated and saved to fig_interpolation.pdf.
        To accomplish this, the input PVA is linearly interpolated, so points between the timestamps of the input pva will have errors.

    Returns
    -------
    Tuple of
        - Nx9 array of interpolated data with the headers of  [lat, lon, alt, vn, ve, vd, r, p, y]
        - length N array of timestamps for each PVA in the prior array
    """
    _vel_t = vel_t if vel_t is not None else llh_t

    t_start_max: float = max([llh_t[0, 0], _vel_t[0, 0], rpy_t[0, 0]])
    t_start_min: float = min([llh_t[0, 0], _vel_t[0, 0], rpy_t[0, 0]])
    t_stop_max: float = max([llh_t[-1, 0], _vel_t[-1, 0], rpy_t[-1, 0]])
    t_stop_min: float = min([llh_t[-1, 0], _vel_t[-1, 0], rpy_t[-1, 0]])

    if t_start is None:
        t_start = t_start_max
    if t_stop is None:
        t_stop = t_stop_min
    if t_start < t_start_min or t_stop > t_stop_max:
        raise ValueError(
            't_start and t_stop must be within time bounds of data provided: \n',
            '\tData start time:\t',
            t_start_max,
            '\n',
            '\tData stop time:\t',
            t_stop_min,
        )

    if (
        interp_type is None
    ):  # None can be handled from arg parse rather than setting a default value above.
        interp_type = 'linear'
    elif interp_type == 'spline':
        if s is None:
            s = [None] * 9
        if w is None:
            w = [None] * 9

    print(
        f'Interpolating PVA\n'
        f'\tStart Time:\t{t_start}\n'
        f'\tStop Time: \t{t_stop}\n'
        f'\tdt:        \t{dt}'
    )

    # Generate time stamps for each step.
    # Times are shifted to start at zero to avoid decimal inefficiencies with large floats
    t_out: np.ndarray = np.arange(
        0, t_stop - t_start + dt, step=dt
    )  # Note: This will drop last point if not equally incremented
    # make sure our t_out does not go beyond the time difference and if so clip out the bad values
    t_out = t_out[t_out <= t_stop - t_start]

    if vel_t is not None:
        t_in = np.array(
            [llh_t[:, 0] - t_start] * 3
            + [vel_t[:, 0] - t_start] * 3
            + [rpy_t[:, 0] - t_start] * 3
        )
        pva_in = np.concatenate((llh_t[:, 1:], vel_t[:, 1:], rpy_t[:, 1:]), axis=1)
        _range1 = 6
    else:
        t_in = np.array([llh_t[:, 0] - t_start] * 3 + [rpy_t[:, 0] - t_start] * 3)
        pva_in = np.concatenate((llh_t[:, 1:], rpy_t[:, 1:]), axis=1)
        _range1 = 3
    _range2 = _range1 + 3

    pva_out = np.empty((len(t_out), _range2))

    # interpolate each column (lat, lon, alt, vel_n, vel_e, vel_d) independently.
    for idx in range(_range1):
        if interp_type != 'spline':
            f = interp1d(
                t_in[idx], pva_in[:, idx], kind=interp_type, assume_sorted=True
            )
            pva_out[:, idx] = f(t_out)
        else:
            f = splrep(t_in[idx], pva_in[:, idx], w=w[idx], s=s[idx])
            pva_out[:, idx] = splev(t_out, f)

    # Interpolate each attitude column (roll, pitch, yaw) independently. separate to handle wrapping
    for idx in range(_range1, _range2):
        if interp_type != 'spline':
            f = interp1d(
                t_in[idx],
                np.unwrap(pva_in[:, idx]),
                kind=interp_type,
                assume_sorted=True,
            )
            pva_out[:, idx] = (np.mod(f(t_out), 2 * np.pi) + np.pi) % (
                2 * np.pi
            ) - np.pi
        else:
            f = splrep(t_in[idx], np.unwrap(pva_in[:, idx]), s=s[idx])
            pva_out[:, idx] = (np.mod(splev(t_out, f), 2 * np.pi) + np.pi) % (
                2 * np.pi
            ) - np.pi

    # shift timestamps back to input start time
    return pva_out, t_out + t_start


def compute_shift(
    truth_time: np.ndarray,
    truth_y: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    # Create an interpolating function for y(x)
    y_interp = interp1d(time, y, kind='linear', fill_value='extrapolate')

    # Define error function to minimize
    def error_function(shift):
        y_shifted = y_interp(truth_time + shift)  # Evaluate y at shifted truth_time
        return np.sum((truth_y - y_shifted) ** 2)  # Compute sum of squared errors

    # Optimize Δx using scalar minimization
    result = minimize_scalar(
        error_function,
        bounds=(-np.ptp(truth_time), np.ptp(truth_time)),
        method='bounded',
    )

    return result.x


def downsample_imu(
    times: np.ndarray, accel: np.ndarray, gyro: np.ndarray, interval=1.0
):
    """Compute average angular rate of each `interval` second sliding window of
    integrated IMU measurements.

    For each window, measurements are summed to obtain a total integrated measurement,
    then divided by the total time interval to compute an overall value for the given
    interval.

    Args:
        times: Sorted array of timestamps.
        accel: Nx3 array of accel measurements corresponding to the timestamps in times,
            where N is the number of time steps.
        gyro: Nx3 array of gyro measurements corresponding to the timestamps in times,
            where N is the number of time steps.
        interval: Interval (seconds) for each window. Defaults to 1.0
    """
    window_start = times[0]
    running_accel_sum = np.zeros(3)
    running_gyro_sum = np.zeros(3)
    out_times = []
    out_accel = []
    out_gyro = []
    for time, a, g in zip(times, accel, gyro):
        cur_interval = time - window_start

        if cur_interval > interval:
            # Completed window, save off average rate
            out_times.append(window_start + cur_interval / 2)
            out_accel.append(running_accel_sum / cur_interval)
            out_gyro.append(running_gyro_sum / cur_interval)
            window_start = time
            running_accel_sum[:] = 0.0
            running_gyro_sum[:] = 0.0

        running_accel_sum += a
        running_gyro_sum += g

    return np.array(out_times), np.array(out_accel), np.array(out_gyro)
