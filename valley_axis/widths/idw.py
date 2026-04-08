import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def compute_idw(centerline_array, mask_array, pixel_size, k=5, power=2):
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
