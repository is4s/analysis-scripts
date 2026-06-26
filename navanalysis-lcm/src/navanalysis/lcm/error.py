import numpy as np
from navtk.navutils import dcm_to_rpy, rpy_to_dcm


def calc_drms(x_err: np.ndarray, y_err: np.ndarray) -> float:
    """
    Compute the Distance Root Mean Square (DRMS) of 2D errors.

    The computation is:
        DRMS = sqrt(mean(x_err^2 + y_err^2))

    Only finite (non-NaN, non-infinite) samples are included in the calculation.

    Args:
        x_err (np.ndarray): N-length array containing error values in the x direction.
        y_err (np.ndarray): N-length array containing error values in the y direction.

    Returns:
        float: The DRMS value in the same units as the input errors.
    """
    squared_errors = x_err**2 + y_err**2
    N = np.count_nonzero(np.isfinite(squared_errors))
    mean_squared_error = np.nansum(squared_errors) / N
    return np.sqrt(mean_squared_error)


def calc_tilts(rpy1: np.ndarray, rpy2: np.ndarray):
    """
    Compute relative tilt angles between two attitude sequences using RPY representation.

    For each time step, this function computes the relative rotation between two
    attitudes specified in roll-pitch-yaw (RPY) form, and expresses that rotation
    again as RPY angles.

    The relative rotation is defined as:
        C_err = C1 @ C2.T

    where C1 and C2 are the DCMs corresponding to `rpy1` and `rpy2`, respectively.
    The resulting RPY angles represent a sequence of rotations (based on the
    convention used in `dcm_to_rpy`) that maps attitude `rpy2` into `rpy1`.

    Args:
        rpy1 (np.ndarray): Nx3 array containing roll, pitch, yaw
            angles (in radians) for the first attitude sequence.
        rpy2 (np.ndarray): Nx3 array containing roll, pitch, yaw
            angles (in radians) for the second attitude sequence.

    Returns:
        np.ndarray: Nx3 array containing the relative RPY angles
        (in radians) representing the rotation from `rpy2` to `rpy1`. Entries
        are set to NaN where either input contains NaNs at a given time step.
    """
    tilts = np.zeros(rpy1.shape)
    for k in range(rpy1.shape[0]):
        if np.isnan(rpy1[k]).any() or np.isnan(rpy2[k]).any():
            tilts[k, :] = np.array([np.nan, np.nan, np.nan])
        else:
            tilts[k, :] = dcm_to_rpy(rpy_to_dcm(rpy1[k, :]) @ rpy_to_dcm(rpy2[k, :]).T)
    return tilts
