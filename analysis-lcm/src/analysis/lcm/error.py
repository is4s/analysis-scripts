import numpy as np
from navtk.navutils import dcm_to_rpy, rpy_to_dcm


def calc_drms(x_err: np.ndarray, y_err: np.ndarray) -> float:
    squared_errors = x_err**2 + y_err**2
    N = np.count_nonzero(np.isfinite(squared_errors))
    mean_squared_error = np.nansum(squared_errors) / N
    return np.sqrt(mean_squared_error)


def calc_tilts(rpy1: np.ndarray, rpy2: np.ndarray):
    """
    Both sources nx3
    """
    tilts = np.zeros(rpy1.shape)
    for k in range(rpy1.shape[0]):
        if np.isnan(rpy1[k]).any() or np.isnan(rpy2[k]).any():
            tilts[k, :] = np.array([np.nan, np.nan, np.nan])
        else:
            tilts[k, :] = dcm_to_rpy(rpy_to_dcm(rpy1[k, :]).T @ rpy_to_dcm(rpy2[k, :]))
    return tilts
