import numpy as np
from navtk.navutils import (
    delta_lat_to_north,
    delta_lon_to_east,
    east_to_delta_lon,
    north_to_delta_lat,
)


def llh_to_ned(llh: np.ndarray, llh0=None):
    out = np.zeros(llh.shape)

    if llh.shape[0] == 0:
        return out

    if llh0 is None:
        llh0 = llh[0]
    lat0, lon0, alt0 = llh0

    for k in range(0, llh.shape[0]):
        out[k, 0] = delta_lat_to_north(llh[k, 0] - lat0, lat0, alt0)
        out[k, 1] = delta_lon_to_east(llh[k, 1] - lon0, lat0, alt0)
        out[k, 2] = alt0 - llh[k, 2]
    return out


def ned_sigma_to_llh_sigma(ned_sigma: np.ndarray, llh: np.ndarray):
    """
    Requires nx3 matrices
    """
    out_sigma = ned_sigma.copy()  # start w/ ned_sigma since vertical sigma is the same
    for k in range(0, ned_sigma.shape[0]):
        out_sigma[k, 0] = north_to_delta_lat(ned_sigma[k, 0], llh[k, 0], llh[k, 2])
        out_sigma[k, 1] = east_to_delta_lon(ned_sigma[k, 1], llh[k, 0], llh[k, 2])
    return out_sigma


def pressure_to_alt(pressure, deg_k=288.15, ref_pressure=101325.0, ref_alt=0.0):
    alt = -(deg_k / 0.0065) * (
        pow(pressure / ref_pressure, 8314.32 * 0.0065 / (9.80665 * 28.9644)) - 1.0
    )
    return alt + ref_alt
