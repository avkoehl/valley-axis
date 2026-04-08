# options IDW, Voronoi allocation, Laplace Harmonic Interpolation
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree


def compute_widths_idw(centerline_array, mask_array, pixel_size, k=5, power=2):
    """Computes continuous valley widths using IDW from the centerline distances."""
    # 1. Distance transform to find valley radius (distance to nearest 0)
    # edt treats 0 as background, so we ensure the mask is strictly boolean
    radius_pixels = distance_transform_edt(mask_array == 1)
    widths_physical = radius_pixels * pixel_size * 2

    # 2. Extract known widths exactly at the centerline pixels
    cy, cx = np.nonzero(centerline_array > 0)
    known_widths = widths_physical[cy, cx]

    # Build KDTree with the centerline pixel coordinates
    tree = cKDTree(np.column_stack((cy, cx)))

    # 3. Get all target pixels in the valley mask we need to interpolate
    ty, tx = np.nonzero(mask_array == 1)
    target_points = np.column_stack((ty, tx))

    # 4. Query the k=5 nearest centerline neighbors for every valid valley pixel
    distances, indices = tree.query(target_points, k=k)

    # 5. Inverse Distance Weighting (IDW)
    # Add a tiny epsilon to prevent division by zero for pixels exactly on the centerline
    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / (distances**power)

    interpolated_widths = np.sum(weights * known_widths[indices], axis=1) / np.sum(
        weights, axis=1
    )

    # Hard-assign the exact width for pixels that lie perfectly on the centerline
    exact_mask = distances[:, 0] <= 1e-12
    interpolated_widths[exact_mask] = known_widths[indices[exact_mask, 0]]

    # 6. Reconstruct the 2D raster array
    out_widths = np.full(mask_array.shape, np.nan, dtype=np.float32)
    out_widths[ty, tx] = interpolated_widths

    return out_widths


def compute_widths_voronoi(centerline_array, mask_array, pixel_size, smooth_sigma=3.0):
    """
    Computes continuous valley widths using Voronoi Allocation
    followed by masked Gaussian smoothing.
    """
    # 1. Exact widths everywhere (2 * distance to nearest boundary)
    radius_pixels = distance_transform_edt(mask_array == 1)
    exact_widths = radius_pixels * pixel_size * 2

    # 2. Nearest-Centerline Allocation (Voronoi projection)
    # distance_transform_edt with return_indices=True gives the coordinates
    # of the absolute nearest zero-pixel. We invert the centerline to target it.
    _, indices = distance_transform_edt(centerline_array == 0, return_indices=True)

    # Map the exact centerline widths outward to the valley walls
    nearest_y, nearest_x = indices[0], indices[1]
    allocated_widths = exact_widths[nearest_y, nearest_x]

    # Isolate to the valley floor
    valley_widths = np.where(mask_array == 1, allocated_widths, 0.0)

    # 3. Masked Gaussian Smoothing
    # To prevent the outside 0s from "dimming" the edges during the blur,
    # we blur the values and the mask separately, then divide them.
    blurred_widths = gaussian_filter(valley_widths, sigma=smooth_sigma)
    blurred_mask = gaussian_filter((mask_array == 1).astype(float), sigma=smooth_sigma)

    smoothed_widths = np.zeros_like(valley_widths)
    valid = blurred_mask > 0.01  # Prevent divide-by-zero
    smoothed_widths[valid] = blurred_widths[valid] / blurred_mask[valid]

    # 4. Re-enforce the mathematically exact widths directly on the centerline pixels
    final_widths = np.where(centerline_array > 0, exact_widths, smoothed_widths)

    # Clean up any stray values outside the mask
    return np.where(mask_array == 1, final_widths, 0.0)
