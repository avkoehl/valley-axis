import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt
import skfmm


def compute_path_segmentation(
    centerline_array: np.ndarray,
    mask_array: np.ndarray,
    segment_to_path: dict[int, int],
) -> np.ndarray:
    """
    Partitions a mask into hierarchical segments based on ordered centerlines,
    using geodesic expansion to prevent overlapping claims.
    """
    path_map = np.full(mask_array.shape, -1, dtype=np.int32)
    claimed_mask = np.zeros_like(mask_array, dtype=bool)

    # Group segment_ids by path_id.
    # Processing lower path_ids first establishes the hierarchy.
    path_to_segments: dict[int, list[int]] = {}
    for seg_id, path_id in segment_to_path.items():
        path_to_segments.setdefault(path_id, []).append(seg_id)

    # Step 1: Global Thickness Estimation
    # Defines the maximum allowed expansion radius at every pixel
    radius_map = distance_transform_edt(mask_array == 1)

    # Step 2: Hierarchical Geodesic Expansion
    for path_id in sorted(path_to_segments.keys()):
        segment_ids = path_to_segments[path_id]
        path_mask = np.isin(centerline_array, segment_ids)

        if not np.any(path_mask):
            continue

        # 2a. Define the Navigable Space
        # Centerline is 0.0, everything else is 1.0
        phi = np.ones_like(mask_array, dtype=float)
        phi[path_mask] = 0.0

        # Background and already claimed territories act as impassable walls
        obstacles = (mask_array != 1) | claimed_mask
        phi_masked = ma.MaskedArray(phi, mask=obstacles)

        # 2b. Geodesic Distance & Value Propagation
        # dist_to_path calculates how far the "water" flowed.
        # extended_radii pushes the local radius limit outward along that flow.
        try:
            dist_to_path, extended_radii = skfmm.extension_velocities(
                phi_masked, radius_map
            )
        except ValueError:
            # Failsafe: if the centerline is completely walled off, skfmm can raise an error.
            continue

        # 2c. Thresholding (The Claim Check)
        valid_claim = (
            (mask_array == 1)
            & (~claimed_mask)
            & (~dist_to_path.mask)  # Ensure the pixel was geodesically reachable
            & (dist_to_path.data <= extended_radii.data)
        )

        # 2d. Commitment
        path_map[valid_claim] = path_id
        claimed_mask[valid_claim] = True

    # Step 3 & 4: Orphan Resolution (Nearest-Neighbor Fallback)
    # Catch valid pixels left behind by sharp corners or radius irregularities
    unclaimed = (mask_array == 1) & (~claimed_mask)
    if np.any(unclaimed):
        # We only want to search against pixels that successfully claimed territory
        search_mask = claimed_mask & (mask_array == 1)

        # We use EDT here because we just need a strict local Voronoi assignment
        # over tiny 1-2 pixel gaps strictly within the remaining mask.
        _, closest_idx = distance_transform_edt(~search_mask, return_indices=True)
        nearest_r, nearest_c = closest_idx[0], closest_idx[1]

        # Assign orphans to the physically closest locked-in territory
        path_map[unclaimed] = path_map[nearest_r[unclaimed], nearest_c[unclaimed]]

    return path_map
