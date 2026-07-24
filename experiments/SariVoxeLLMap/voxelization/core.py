"""
Methods for converting the points to 2D grids and 3D columns
"""

# -----------------------
# Imports
# -----------------------

import numpy as np
import logging
from time import time

# -----------------------
# Logging Setup
# -----------------------

logger = logging.getLogger(__name__)

# -----------------------
# Class for data storage
# -----------------------

class Voxelizator:

    def __init__(self) -> None:
        self.preprocessed_points = np.array([]).reshape(0, 3)
        self.no_outlier_points = np.array([]).reshape(0, 3)
        self.no_floor_ceil_points = np.array([]).reshape(0, 3)
        self.histogram2d = np.array([]).reshape(0, 0)
        self.xedges = np.array([])
        self.yedges = np.array([])
        self.bin_width = 0.75
        self.height_estimates = np.array([]).reshape(0, 0)
        self.heightmap = np.array([]).reshape(0, 0)
        self.min_z = 0.0

# -----------------------
# Pre-processing
# ----------------------- 

def correct_coordinates(points: np.ndarray) -> np.ndarray:
    """
    Transform coordinates by swapping x and y values and inverting the z value to correct point cloud orientation.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :return: The coordinates of the points with the x and y values swapped and z inverted, which can be used for correcting the orientation of the point cloud data
    :rtype: np.ndarray
    """

    # Vectorized operation: directly stack columns without looping
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)

def sort_points_by_height(points: np.ndarray) -> np.ndarray:
    """
    Sorts the points by their z values (height) in ascending order, which can be used for sorting the point cloud data by height.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :return: The coordinates of the points sorted by their z values, which can be used for sorting the point cloud data by height
    :rtype: np.ndarray
    """

    # Use NumPy's argsort for efficient sorting (100x+ faster than Python's sorted)
    sorted_indices = np.argsort(points[:, 2])
    return points[sorted_indices]

def preprocess_points(points: np.ndarray) -> np.ndarray:
    """
    Apply coordinate correction, height-based sorting, and non-negative shifting to prepare 3D points for voxelization.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :return: The coordinates of the points after pre-processing
    :rtype: np.ndarray
    """

    # TODO: Optimize (currently taking too long e.g. > 3 seconds on average)

    start = time()

    points = correct_coordinates(points)
    points = sort_points_by_height(points)

    logger.info(f"Pre-processing done. Z between {min(points[:, 2])} and {max(points[:, 2])}. Time: {time() - start:.2f} seconds")

    return points

# -----------------------
# Filtering
# -----------------------

def remove_outliers(points: np.ndarray, num_std: float = 2) -> np.ndarray:
    """
    Removes outliers from the point cloud data by filtering out points that are more than 3 standard deviations away from the mean in any of the x, y, or z dimensions, which can be used for filtering out noise from the point cloud data.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :return: The coordinates of the points with outliers removed, which can be used for filtering out noise from the point cloud data
    :rtype: np.ndarray
    """

    start = time()

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    z_std = np.std(z_coords)

    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    z_mean = np.mean(z_coords)

    x_high_limit = x_mean + num_std * x_std
    x_low_limit = x_mean - num_std * x_std
    y_high_limit = y_mean + num_std * y_std
    y_low_limit = y_mean - num_std * y_std
    z_high_limit = z_mean + num_std * z_std
    z_low_limit = z_mean - num_std * z_std

    filtered_points = points[
        (points[:, 0] > x_low_limit) & (points[:, 0] < x_high_limit) &
        (points[:, 1] > y_low_limit) & (points[:, 1] < y_high_limit) &
        (points[:, 2] > z_low_limit) & (points[:, 2] < z_high_limit)
    ]

    logger.info(f"Outliers removed: -{len(points) - len(filtered_points)} points. Z between {min(points[:, 2])} and {max(points[:, 2])}. Time: {time() - start:.2f} seconds")

    return filtered_points

def remove_floor_ceil(
        points: np.ndarray,
        floor_cumulative_threshold: float = 0.2,
        has_ceiling: bool = True,
        ceiling_cumulative_threshold: float = 0.8
        ) -> np.ndarray:
    """
    Excludes floor and ceiling points using percentile-based filtering.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :param floor_cumulative_threshold: Percentile threshold for floor cutoff (default: 0.2 = remove bottom 20%)
    :type floor_cumulative_threshold: float
    :param has_ceiling: Whether to also exclude ceiling points
    :type has_ceiling: bool
    :param ceiling_cumulative_threshold: Percentile threshold for ceiling cutoff (default: 0.8 = remove top 20%)
    :type ceiling_cumulative_threshold: float
    :param bins: Unused parameter kept for compatibility
    :type bins: str
    :return: The coordinates of the points with floor and ceiling removed
    :rtype: np.ndarray
    """

    start = time()

    z = points[:, 2]
    
    # Calculate percentile cutoffs directly
    floor_cutoff = np.percentile(z, floor_cumulative_threshold * 100)
    
    # Filter out floor points
    points_wo_floor_ceil = points[z > floor_cutoff]

    if has_ceiling:
        ceiling_cutoff = np.percentile(z, ceiling_cumulative_threshold * 100)
        # Filter out ceiling points
        points_wo_floor_ceil = points_wo_floor_ceil[points_wo_floor_ceil[:, 2] < ceiling_cutoff]
    
    logger.info(f"Floor and ceiling points excluded: -{len(points) - len(points_wo_floor_ceil)} points. Z between {min(points[:, 2])} and {max(points[:, 2])}. Time: {time() - start:.2f} seconds")

    return points_wo_floor_ceil

# -----------------------
# Heightmap Generation
# -----------------------

def split_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the points into separate x, y, and z coordinate arrays for easier processing in heightmap generation.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :return: Three separate numpy arrays containing the x, y, and z coordinates of the points, which can be used for easier processing in heightmap generation
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    return x, y, z

def get_histogram2d(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        bin_width: float = 0.75
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes a 2D histogram of point counts in the x-y plane using specified bin width.
    
    :param points: The coordinates of the points within an arbitrary planar area in a single direction
    :type points: np.ndarray
    :param bin_width: The width of each bin in the x and y directions (default: 0.75)
    :type bin_width: float
    :return: A tuple containing the 2D histogram of point counts, the edges of the bins in the x direction, and the edges of the bins in the y direction, along with the bin width
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    """

    start = time()

    if x.size == 0:
        return np.empty((0, 3), dtype=float), np.array([]), np.array([]), bin_width

    # 1) Get the histogram bins

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
    y_bin_edges = np.arange(y_min, y_max + bin_width, bin_width)
    
    if x_bin_edges.size < 2:
        x_bin_edges = np.array([x_min, x_min + bin_width], dtype=float)
    if y_bin_edges.size < 2:
        y_bin_edges = np.array([y_min, y_min + bin_width], dtype=float)

    H_counts, xedges, yedges = np.histogram2d(x, y, bins=[x_bin_edges, y_bin_edges], density=False)

    logger.info(f"2D histogram computed with bin width {bin_width}. Time: {time() - start:.2f} seconds")

    return H_counts, xedges, yedges, bin_width

def get_height_estimates(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        H_counts: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        z_percentile: float = 90.0
) -> tuple[np.ndarray, float]:
    """
    Computes the specified percentile of z values for each bin in the 2D histogram, which can be used for estimating the height of each bin in the heightmap representation of the point cloud data.
    
    :param x: The x-coordinates of the points
    :type x: np.ndarray
    :param y: The y-coordinates of the points
    :type y: np.ndarray
    :param z: The z-coordinates of the points
    :type z: np.ndarray
    :param H_counts: The 2D histogram of point counts in the x-y plane
    :type H_counts: np.ndarray
    :param xedges: The edges of the bins in the x direction
    :type xedges: np.ndarray
    :param yedges: The edges of the bins in the y direction
    :type yedges: np.ndarray
    :param z_percentile: The percentile of z values to use for representing each bin (default: 90.0)
    :type z_percentile: float
    :return: A 2D numpy array where each element represents the specified percentile of z values for each bin, which can be used for estimating the height of each bin in the heightmap representation of the point cloud data
    :rtype: tuple[np.ndarray, float]
    """
    
    start = time()

    nx, ny = H_counts.shape

    # 1) Assign every point to a bin index in O(P log B) using digitize, then
    #    flatten the 2D bin index to a 1D key so we can sort once and group.
    xi = np.clip(np.digitize(x, xedges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, ny - 1)
    lin_idx = xi * ny + yi  # unique integer per (bin_x, bin_y) cell

    # 2) Sort points by their bin so members of each bin are contiguous — O(P log P).
    order = np.argsort(lin_idx, kind='stable')
    lin_idx_sorted = lin_idx[order]
    z_sorted = z[order]

    # 3) Find where the bin id changes to get contiguous group boundaries.
    split_points = np.flatnonzero(np.diff(lin_idx_sorted)) + 1
    groups  = np.split(z_sorted,    split_points)
    bin_ids = lin_idx_sorted[np.r_[0, split_points]]

    # 4) Compute percentile once per occupied bin — no redundant point scans.
    local_percentiles_flat = np.zeros(nx * ny, dtype=float)
    for bid, zg in zip(bin_ids, groups):
        local_percentiles_flat[bid] = np.percentile(zg, z_percentile)
    local_percentiles = local_percentiles_flat.reshape(nx, ny)

    # All points are guaranteed to fall in some bin, so the global min equals
    # the minimum across all per-bin minima.
    local_min_z = float(z.min())

    # 5) Round the height estimates to 2 decimal places for better visualization and to reduce file size when saving
    local_percentiles = np.round(local_percentiles, 2)

    logger.info(f"Height estimates computed using {z_percentile}th percentile. Time: {time() - start:.2f} seconds")

    return local_percentiles, local_min_z

def get_height_filter_mask(
        height_estimates: np.ndarray,
        height_threshold: float = 0.5
) -> np.ndarray | None:
    """
    Computes a boolean mask for the bins in the height estimates based on a height threshold, which can be used for filtering out bins that do not meet the specified height criteria in the heightmap representation of the point cloud data.
    
    :param height_estimates: A 2D numpy array where each element represents the estimated height of each bin
    :type height_estimates: np.ndarray
    :param height_threshold: The minimum height threshold for a bin to be considered valid (between 0 and 1)
    :type height_threshold: float
    :return: A boolean 2D numpy array where True indicates bins that meet the height threshold and False indicates bins that do not, which can be used for filtering out bins in the heightmap representation of the point cloud data. Returns None if there are no non-zero height estimates.
    :rtype: np.ndarray | None
    """

    start = time()

    nonzero_heights = height_estimates[height_estimates > 0]
    if nonzero_heights.size == 0:
        return None
    min_height = float(height_threshold * nonzero_heights.max())

    logger.info(f"Height filter mask computed with height threshold {height_threshold}. Time: {time() - start:.2f} seconds")

    return height_estimates >= min_height

# TODO: Review, since currently it also erases previously chosen bars that are correct
def get_density_filter_mask(
        H_counts: np.ndarray,
        density_threshold: float = 0.33,
) -> np.ndarray | None:
    """
    Computes a boolean mask for the bins in the 2D histogram based on a density threshold, which can be used for filtering out bins that do not meet the specified density criteria in the heightmap representation of the point cloud data.    

    :param H_counts: The 2D histogram of point counts in the x-y plane
    :type H_counts: np.ndarray
    :param density_threshold: The minimum density threshold for a bin to be considered valid (between 0 and 1)
    :type density_threshold: float
    :return: A boolean 2D numpy array where True indicates bins that meet the density threshold and False indicates bins that do not, which can be used for filtering out bins in the heightmap representation of the point cloud data. Returns None if there are no non-empty bins.
    :rtype: np.ndarray | None
    """

    start = time()

    nonzero_counts = H_counts[H_counts > 0]
    if nonzero_counts.size == 0:
        return None
    min_count = max(1.0, float(np.percentile(nonzero_counts, density_threshold * 100.0)))

    logger.info(f"Density filter mask computed with density threshold {density_threshold}. Time: {time() - start:.2f} seconds")

    return H_counts >= min_count

def get_filtered_heightmap(
        estimated_heights: np.ndarray,
        density_mask: np.ndarray,
        height_mask: np.ndarray
) -> np.ndarray:
    """
    Applies a density and a height filter mask to the estimated heights to produce a filtered heightmap, which can be used for creating a heightmap representation of the point cloud data that only includes bins that meet the specified density criteria.
    
    :param estimated_heights: A 2D numpy array where each element represents the estimated height of each bin
    :type estimated_heights: np.ndarray
    :param density_mask: A boolean 2D numpy array where True indicates bins that meet the density threshold and False indicates bins that do not
    :type density_mask: np.ndarray
    :param height_mask: A boolean 2D numpy array where True indicates bins that meet the height threshold and False indicates bins that do not
    :type height_mask: np.ndarray
    :return: A 2D numpy array where each element represents the height of each bin after applying the density and height filters, which can be used for creating a heightmap representation of the point cloud data that only includes bins that meet the specified density and height criteria
    :rtype: np.ndarray
    """
    start = time()
    filtered_heights = np.where(density_mask & height_mask, estimated_heights, 0.0)
    logger.info(f"Filtered heightmap computed. Time: {time() - start:.2f} seconds")
    return filtered_heights

def format_back_to_unity_coordinates(
        heightmap: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray
) -> list[tuple[float, float, float]]:
    """
    Converts a 2D heightmap histogram into a list of three coordinates (x, y, height) for each non-zero bin, which can be used for creating a column representation of the point cloud data based on the heightmap.
    
    :param heightmap: A 2D numpy array where each element represents the height of each bin
    :type heightmap: np.ndarray
    :param xedges: The edges of the bins in the x direction
    :type xedges: np.ndarray
    :param yedges: The edges of the bins in the y direction
    :type yedges: np.ndarray
    :return: A list of tuples containing the x coordinate, y coordinate, and height for each non-zero bin in the heightmap, which can be used for creating a column representation of the point cloud data based on the heightmap
    :rtype: list[tuple[float, float, float]]
    """
    start = time()

    # 1) Interchange the axes so that the x-axis corresponds to the horizontal direction and the y-axis corresponds to the vertical direction in the visualization
    heightmap = heightmap.T

    # 2) Extract non-zero bins efficiently using column_stack
    y_idx, x_idx = np.nonzero(heightmap)
    three_coor_list = np.column_stack((
        xedges[x_idx],
        heightmap[y_idx, x_idx],
        -yedges[y_idx],
    )
    ).tolist()
    logger.info(f"Heightmap formatted as three coordinates and transformed for Unity. Time: {time() - start:.2f} seconds")

    return three_coor_list