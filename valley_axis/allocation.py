import numpy as np
import numpy.ma as ma
import xarray as xr
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import skfmm

from .centerlines import Centerlines


def get_allocation(centerlines, mask):
    segment_raster = centerlines.label_by_segment()
    segment_array = segment_raster.values
    mask_array = mask.values == 1

    allocation = np.zeros(mask_array.shape, dtype=np.uint32)
    claimed = np.zeros_like(mask_array, dtype=bool)
    radius_map = distance_transform_edt(mask_array)

    # Process tier by tier
    segs = centerlines.segments.sort_values(["path_label", "segment_id"])
    for path_label, group in segs.groupby("path_label", sort=True):
        tier_seg_ids = group["segment_id"].tolist()
        tier_seeds = np.isin(segment_array, tier_seg_ids) & ~claimed
        if not tier_seeds.any():
            continue

        phi = np.ones_like(mask_array, dtype=float)
        phi[tier_seeds] = 0.0
        obstacles = (~mask_array) | claimed
        phi_masked = ma.MaskedArray(phi, mask=obstacles)

        try:
            dist, ext_radius = skfmm.extension_velocities(phi_masked, radius_map)
        except ValueError:
            continue

        claim = mask_array & ~claimed & ~dist.mask & (dist.data <= ext_radius.data)

        if not claim.any():
            continue

        # Crop for the backfill EDT
        rows, cols = np.where(claim)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        sl = (slice(r0, r1), slice(c0, c1))

        local_tier_seeds = tier_seeds[sl]
        local_segment_array = segment_array[sl]
        local_claim = claim[sl]

        # Nearest tier-seed in Euclidean distance (within the path's bbox)
        _, inds = distance_transform_edt(~local_tier_seeds, return_indices=True)
        nearest_seg = local_segment_array[inds[0], inds[1]]

        allocation[sl][local_claim] = nearest_seg[local_claim]
        claimed[sl][local_claim] = True

    # Orphan resolution unchanged
    unclaimed = mask_array & ~claimed
    if unclaimed.any() and claimed.any():
        dist_from_claimed = distance_transform_edt(~claimed)
        allocation = watershed(
            image=dist_from_claimed, markers=allocation, mask=mask_array
        ).astype(np.uint32)

    out = xr.DataArray(allocation, coords=mask.coords, dims=mask.dims, attrs=mask.attrs)
    out.rio.write_crs(mask.rio.crs, inplace=True)
    out.rio.write_transform(mask.rio.transform(), inplace=True)
    return out
