from lib import trace, grid_intersect
import numpy as np


def take_measurement(n_grid: np.int64, n_rays: np.int64, r_theta: np.float64):
    """
    Take a measurement with the tomograph from direction r_dir

    Arguments:
    n_grid  : number of cells of grid in each direction
    n_rays  : number of parallel rays
    r_theta : direction of rays (in radians)

    Return:
    intensities : measured intensities for rays (ndarray)
    idx         : indices of rays (ndarray)
    idx_isect   : indices of intersected cells (ndarray)
    dt          : lengths of segments in intersected cells (ndarray)

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # compute ray direction in Cartesian coordinates
    cs = np.cos(r_theta)
    sn = np.sin(r_theta)
    r_dir = np.array([-cs, -sn])

    # compute start positions for rays
    r_pos = np.zeros((n_rays, 2))
    for i, g in enumerate(np.linspace(-0.99, 0.99, n_rays)):
        r_pos[i] = np.array([cs - sn * g, sn + cs * g])
    else:
        r_pos[0] = np.array([cs, sn])

    # compute measures intensities for each ray
    intensities = np.zeros(n_rays)
    for i, rs in enumerate(r_pos):
        intensities[i] = trace(rs, r_dir)
    # take exponential fall off into account
    intensities = np.log(1.0 / intensities)

    # compute traversal distance in each grid cell
    idx, idx_isect, dt = grid_intersect(n_grid, r_pos, r_dir)

    return intensities, idx, idx_isect, dt
