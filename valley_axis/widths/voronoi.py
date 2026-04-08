import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def compute_voronoi(centerline_array, mask_array, pixel_size, smooth_sigma=3.0):
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
