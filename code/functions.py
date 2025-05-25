import numpy as np
from scipy.spatial import distance


__all__ = [
    "generate_cluster", "generate_random_background", "mask_void_region",
    "generate_filament", "add_shapes",
    "compute_2pcf", "compute_tangential_and_cross",
    "bin_2pcf", "compute_2pcf_and_xi"
]


def generate_cluster(center, n_points, spread, seed=42):
    """
    Generate a 2D Gaussian cluster of points around a specified center.

    Parameters
    ----------
    center : tuple of float
        The (x, y) coordinates of the cluster center.
    n_points : int
        Number of points to generate.
    spread : float
        Standard deviation (sigma) of the Gaussian spread around the center.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    x : ndarray
        x-coordinates of the generated points.
    y : ndarray
        y-coordinates of the generated points.
    """
    rng = np.random.default_rng(seed)  # Use default_rng for better random number generation
    # Generate points from a normal distribution centered at 'center'
    x = rng.normal(loc=center[0], scale=spread, size=n_points)
    y = rng.normal(loc=center[1], scale=spread, size=n_points)
    return x, y


def generate_random_background(n_points, xlim, ylim, seed=42):
    """
    Generate uniformly distributed 2D background points within a rectangular region.

    Parameters
    ----------
    n_points : int
        Number of random background points to generate.
    xlim : tuple of float
        Minimum and maximum x-values (x_min, x_max).
    ylim : tuple of float
        Minimum and maximum y-values (y_min, y_max).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    x : ndarray
        x-coordinates of the generated background points.
    y : ndarray
        y-coordinates of the generated background points.
    """
    rng = np.random.default_rng(seed)  # Use default_rng for better random number generation
    x = rng.uniform(xlim[0], xlim[1], n_points)  # Generate uniform random x-coordinates
    y = rng.uniform(ylim[0], ylim[1], n_points)  # Generate uniform random y-coordinates
    return x, y

def mask_void_region(x, y, center, radius):
    """
    Mask out points inside a circular void region.

    Parameters
    ----------
    x : ndarray
        x-coordinates of the points.
    y : ndarray
        y-coordinates of the points.
    center : tuple of float
        (x, y) coordinates of the void center.
    radius : float
        Radius of the circular region to mask out.

    Returns
    -------
    x_masked : ndarray
        x-coordinates of points outside the void.
    y_masked : ndarray
        y-coordinates of points outside the void.
    """
    dx = x - center[0]  # Difference from void center in x-direction
    dy = y - center[1]  # Difference from void center in y-direction
    dist = np.sqrt(dx**2 + dy**2)  # Euclidean distance from void center
    mask = dist > radius  # Keep points outside the void
    return x[mask], y[mask]


def generate_filament(start, end, n_points, thickness, seed=42):
    """
    Generate a linear filament of points with Gaussian scatter around a central axis.

    Parameters
    ----------
    start : tuple of float
        (x, y) coordinates of the starting point of the filament.
    end : tuple of float
        (x, y) coordinates of the ending point of the filament.
    n_points : int
        Number of points to generate along the filament.
    thickness : float
        Standard deviation of the Gaussian scatter perpendicular to the filament.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    x : ndarray
        x-coordinates of the generated filament points.
    y : ndarray
        y-coordinates of the generated filament points.
    """
    rng = np.random.default_rng(seed)  # Use default_rng for better random number generation

    # Generate a straight line from start to end
    x_line = np.linspace(start[0], end[0], n_points)
    y_line = np.linspace(start[1], end[1], n_points)

    # Add Gaussian noise to create thickness
    x = x_line + rng.normal(0, thickness, n_points)
    y = y_line + rng.normal(0, thickness, n_points)

    return x, y


def compute_2pcf_pairwise(x, y, bins):
    """
    Compute the two-point correlation function as a histogram of pairwise separations.

    Parameters
    ----------
    x : ndarray
        x-coordinates of the points.
    y : ndarray
        y-coordinates of the points.
    bins : ndarray
        Array of bin edges for separation distances.

    Returns
    -------
    r : ndarray
        Midpoints of the separation bins.
    counts : ndarray
        Number of point pairs in each separation bin.
    """
    # Stack coordinates into shape (N, 2)
    coords = np.vstack([x, y]).T

    # Compute all pairwise Euclidean distances
    dists = distance.pdist(coords)

    # Histogram the distances into bins
    counts, bin_edges = np.histogram(dists, bins=bins)

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bin_centers, counts


def add_shapes(x, shear=(0.0, 0.0), intrinsic_dispersion=0.05):
    """
    Assign galaxy shapes (ellipticities) to points with intrinsic noise and optional shear.

    Parameters
    ----------
    x : ndarray
        x-coordinates of the galaxy positions (used to determine the number of shapes).
    shear : tuple of float, optional
        Constant shear components (gamma1, gamma2) added to all shapes. Default is (0.0, 0.0).
    intrinsic_dispersion : float, optional
        Standard deviation of the intrinsic ellipticity noise. Default is 0.05.

    Returns
    -------
    e1 : ndarray
        First ellipticity component for each galaxy.
    e2 : ndarray
        Second ellipticity component for each galaxy.
    """
    num = len(x)  # Number of galaxies
    # Generate random ellipticities with intrinsic noise
    e1 = np.random.normal(loc=0.0, scale=intrinsic_dispersion, size=num) + shear[0]
    e2 = np.random.normal(loc=0.0, scale=intrinsic_dispersion, size=num) + shear[1]
    return e1, e2


def compute_tangential_and_cross(x, y, e1, e2):
    """
    Compute the pairwise tangential and cross ellipticity correlation products.

    This function computes the tangential (et) and cross (ex) components of the ellipticity
    for all unique galaxy pairs, rotates them into the frame defined by the pair separation,
    and returns the products et_i * et_j and ex_i * ex_j as a function of pair separation.

    Parameters
    ----------
    x : ndarray
        x-coordinates of the galaxies.
    y : ndarray
        y-coordinates of the galaxies.
    e1 : ndarray
        First ellipticity component of the galaxies.
    e2 : ndarray
        Second ellipticity component of the galaxies.

    Returns
    -------
    r : ndarray
        Pairwise separations between galaxies.
    etet : ndarray
        Products of tangential ellipticity components for each pair.
    exex : ndarray
        Products of cross ellipticity components for each pair.
    """
    num = len(x)  # Number of galaxies
    r_list = []
    etet_list = []
    exex_list = []

    for i in range(num):
        for j in range(i + 1, num):
            # Compute separation vector and distance
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            r = np.hypot(dx, dy)
            phi = np.arctan2(dy, dx)

            # Shear rotation angle
            cos2phi = np.cos(2 * phi)
            sin2phi = np.sin(2 * phi)

            # Rotate ellipticity of each galaxy into tangential/cross frame
            et_i = -(e1[i] * cos2phi + e2[i] * sin2phi)
            ex_i = -(e1[i] * sin2phi - e2[i] * cos2phi)
            et_j = -(e1[j] * cos2phi + e2[j] * sin2phi)
            ex_j = -(e1[j] * sin2phi - e2[j] * cos2phi)

            # Store correlation products
            r_list.append(r)
            etet_list.append(et_i * et_j)
            exex_list.append(ex_i * ex_j)

    return np.array(r_list), np.array(etet_list), np.array(exex_list)


def bin_2pcf(r, etprod, exprod, bins):
    """
    Bin the tangential and cross ellipticity products to compute xi+ and xi−.

    Parameters
    ----------
    r : ndarray
        Pairwise separations between galaxies.
    etprod : ndarray
        Products of tangential ellipticity components (et_i * et_j).
    exprod : ndarray
        Products of cross ellipticity components (ex_i * ex_j).
    bins : ndarray
        Array of bin edges for separation.

    Returns
    -------
    r_centers : ndarray
        Midpoints of the separation bins.
    xi_plus : ndarray
        Binned values of xi+ = ⟨et_i * et_j + ex_i * ex_j⟩.
    xi_minus : ndarray
        Binned values of xi− = ⟨et_i * et_j - ex_i * ex_j⟩.
    """
    bin_idx = np.digitize(r, bins)  # Assign each distance to a bin
    # Lists to store xi+ and xi− values and bin centers
    xi_plus = []
    xi_minus = []
    bin_centers = []

    # Loop over bins to compute averages
    for i in range(1, len(bins)):
        mask = bin_idx == i  # Select indices in the current bin
        if np.any(mask):  # Check if there are any points in this bin
            et = etprod[mask]  # Tangential ellipticity products
            ex = exprod[mask]  # Cross ellipticity products
            xi_plus.append(np.mean(et + ex))  # Average xi+
            xi_minus.append(np.mean(et - ex))  # Average xi−
            bin_centers.append(0.5 * (bins[i] + bins[i - 1]))  # Bin center

    return np.array(bin_centers), np.array(xi_plus), np.array(xi_minus)


def compute_2pcf_and_xi(x, y, bins, n_random=5000, seed=42, bounds=None):
    """
    Compute the 2-point correlation function xi(r) using DD/RR - 1.

    Parameters
    ----------
    x : ndarray
        x-coordinates of the data points.
    y : ndarray
        y-coordinates of the data points.
    bins : ndarray
        Array of bin edges for separation distances.
    n_random : int, optional
        Number of points to generate in the random catalog. Default is 5000.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    bounds : tuple of tuple, optional
        ((x_min, x_max), (y_min, y_max)) bounding box for the random catalog.
        If None, bounds are inferred from the data.

    Returns
    -------
    r : ndarray
        Midpoints of the separation bins.
    dd_hist : ndarray
        Raw pair counts (DD) in each bin.
    xi : ndarray
        Estimated correlation function xi(r).
    """
    # Compute data-data pairwise distances
    coords_data = np.vstack([x, y]).T  # Stack x and y coordinates
    dd_dists = distance.pdist(coords_data)  # Compute pairwise distances
    dd_hist, bin_edges = np.histogram(dd_dists, bins=bins)  # Histogram distances

    # Define random bounds if not provided
    if bounds is None:
        xlim = (x.min(), x.max())
        ylim = (y.min(), y.max())
    else:
        xlim, ylim = bounds

    # Generate random catalog within bounds
    rng = np.random.default_rng(seed)  # Use default_rng for better random number generation
    x_rand = rng.uniform(xlim[0], xlim[1], n_random)  # Generate uniform random x-coordinates
    y_rand = rng.uniform(ylim[0], ylim[1], n_random)  # Generate uniform random y-coordinates
    rr_coords = np.vstack([x_rand, y_rand]).T  # Stack random x and y coordinates
    rr_dists = distance.pdist(rr_coords)  # Compute pairwise distances for random points
    rr_hist, _ = np.histogram(rr_dists, bins=bin_edges)  # Histogram distances

    # Normalize and compute xi = DD / RR - 1
    dd_norm = dd_hist / np.sum(dd_hist)
    rr_norm = rr_hist / np.sum(rr_hist)

    # Avoid division by zero and handle invalid values
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.where(rr_norm > 0, dd_norm / rr_norm - 1, 0)

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bin_centers, dd_hist, xi


def assign_realistic_shear(x, y, center=(0, 0), amplitude=0.05, sigma=5.0, mode='tangential', noise_std=0.01):
    """
    Assigns spatially varying shear to mimic radial or tangential alignment.

    Parameters
    ----------
    x, y : ndarray
        Coordinates of galaxies.
    center : tuple
        Center of the shear pattern.
    amplitude : float
        Maximum shear amplitude.
    sigma : float
        Controls spatial decay.
    mode : str
        'radial' or 'tangential' alignment.
    noise_std : float
        Gaussian noise added to mimic shape dispersion.

    Returns
    -------
    e1, e2 : ndarray
        Ellipticity components with spatial shear.
    """
    # Define the shear center and compute distances
    dx = x - center[0]
    dy = y - center[1]
    r2 = dx**2 + dy**2
    # Compute the angle of the shear
    phi = np.arctan2(dy, dx)

    # Determine the angle based on the mode
    if mode == 'tangential':
        angle = phi + np.pi / 2
    elif mode == 'radial':
        angle = phi
    else:
        raise ValueError("mode must be 'tangential' or 'radial'")

    # Now compute the shear components
    # First, compute the shear amplitude
    shear_amp = amplitude * np.exp(-r2 / (2 * sigma**2))
    # Then, compute the shear components
    gamma1 = shear_amp * np.cos(2 * angle)
    gamma2 = shear_amp * np.sin(2 * angle)

    # Finally, obtain the ellipticity components with added gaussian noise
    e1 = gamma1 + np.random.normal(0, noise_std, size=len(x))
    e2 = gamma2 + np.random.normal(0, noise_std, size=len(x))
    return e1, e2
