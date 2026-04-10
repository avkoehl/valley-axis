import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_hierarchical_voronoi(
    centerline_array, mask_array, pixel_size, centerlines_gdf
):
    """
    Assigns a thickness value and source segment ID to valley pixels based on hierarchical centerlines.
    """
    # 1. Global EDT: Distance from every valley pixel to the valley wall
    radius_map = distance_transform_edt(mask_array == 1)
    exact_widths = radius_map * pixel_size * 2.0

    output_thickness = np.zeros_like(mask_array, dtype=np.float32)
    output_sources = np.zeros_like(
        mask_array, dtype=centerline_array.dtype
    )  # New tracking array!
    claimed_mask = np.zeros_like(mask_array, dtype=bool)

    # 2. Extract and sort path labels
    unique_paths = centerlines_gdf["path_label"].unique()

    def get_path_int(label):
        try:
            return int(label.split("_")[1])
        except (IndexError, ValueError):
            return 999999

    sorted_paths = sorted(unique_paths, key=get_path_int)

    # 3. Process top-down
    for path_lbl in sorted_paths:
        path_segments = centerlines_gdf[centerlines_gdf["path_label"] == path_lbl]
        segment_ids = path_segments["segment_id"].values

        if len(segment_ids) == 0:
            continue

        # 4. Create the path mask
        path_mask = np.isin(centerline_array, segment_ids)

        if not np.any(path_mask):
            continue

        # 5. Get distance to this path
        dist_to_path, closest_idx = distance_transform_edt(
            ~path_mask, return_indices=True
        )

        closest_r = closest_idx[0]
        closest_c = closest_idx[1]

        # 6. Look up the variables at the closest centerline pixels
        path_radii_px = radius_map[closest_r, closest_c]
        path_widths_physical = exact_widths[closest_r, closest_c]

        # Look up the actual segment ID at that nearest centerline pixel
        closest_segment_ids = centerline_array[closest_r, closest_c]

        # 7. Define the rules for claiming
        valid_claim = (
            (mask_array == 1) & (~claimed_mask) & (dist_to_path <= path_radii_px)
        )

        # 8. Assign thickness, source ID, and mark claimed
        output_thickness[valid_claim] = path_widths_physical[valid_claim]
        output_sources[valid_claim] = closest_segment_ids[valid_claim]
        claimed_mask[valid_claim] = True

    # --- Gap Fill Logic ---
    unclaimed_valley = (mask_array == 1) & (~claimed_mask)

    if np.any(unclaimed_valley):
        # Calculate distance to the nearest CLAIMED pixel
        _, closest_idx = distance_transform_edt(~claimed_mask, return_indices=True)

        nearest_r = closest_idx[0]
        nearest_c = closest_idx[1]

        # Assign both the thickness AND the source ID of that nearest claimed pixel
        output_thickness[unclaimed_valley] = output_thickness[
            nearest_r[unclaimed_valley], nearest_c[unclaimed_valley]
        ]
        output_sources[unclaimed_valley] = output_sources[
            nearest_r[unclaimed_valley], nearest_c[unclaimed_valley]
        ]
        claimed_mask[unclaimed_valley] = True

    return output_thickness, output_sources
