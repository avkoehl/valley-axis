import numpy as np
import numpy.ma as ma
import xarray as xr
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import skfmm

from .centerlines import Centerlines


def get_allocation(centerlines, mask):
    path_raster = centerlines.label_by_path()
    path_array = path_raster.values
    mask_array = mask.values == 1
    allocation = np.zeros(mask_array.shape, dtype=np.uint32)
    claimed = np.zeros_like(mask_array, dtype=bool)
    radius_map = distance_transform_edt(mask_array)

    # Order by (network_id, path_label) so each network's mainstem is
    # processed before its tributaries. path_uid is the globally-unique
    # integer we actually write into the raster.
    ordered = centerlines.segments[["network_id", "path_label", "path_uid"]].drop_duplicates()
    ordered = ordered.sort_values(["network_id", "path_label"])
    path_uids = ordered["path_uid"].tolist()

    for path_uid in path_uids:
        tier_seeds = (path_array == path_uid) & ~claimed
        if not tier_seeds.any():
            continue
        phi = np.ones_like(mask_array, dtype=float)
        phi[tier_seeds] = 0.0
        obstacles = (~mask_array) | claimed
        phi_masked = ma.MaskedArray(phi, mask=obstacles)
        try:
            dist, ext_radius = skfmm.extension_velocities(
                phi_masked, radius_map, narrow=radius_map.max() + 1
            )
        except ValueError:
            continue
        dist_mask = np.ma.getmaskarray(dist)
        claim = mask_array & ~claimed & ~dist_mask & (dist.data <= ext_radius.data)
        if not claim.any():
            continue
        allocation[claim] = path_uid
        claimed |= claim

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


def subdivide_paths_into_segments(path_allocation, centerlines):
    segment_array = centerlines.label_by_segment().values
    path_alloc_array = path_allocation.values
    segments_df = centerlines.segments

    out_array = np.zeros(path_alloc_array.shape, dtype=np.uint32)

    for path_uid, group in segments_df.groupby("path_uid"):
        territory = path_alloc_array == path_uid
        if not territory.any():
            continue
        seeds = territory & np.isin(segment_array, group["segment_id"].to_numpy())
        if not seeds.any():
            continue
        rows, cols = np.where(territory)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        sl = (slice(r0, r1), slice(c0, c1))
        local_seeds = seeds[sl]
        local_segments = segment_array[sl]
        local_territory = territory[sl]
        _, inds = distance_transform_edt(~local_seeds, return_indices=True)
        nearest = local_segments[inds[0], inds[1]]
        out_array[sl][local_territory] = nearest[local_territory]

    out = xr.DataArray(
        out_array,
        coords=path_allocation.coords,
        dims=path_allocation.dims,
        attrs=path_allocation.attrs,
    )
    out.rio.write_crs(path_allocation.rio.crs, inplace=True)
    out.rio.write_transform(path_allocation.rio.transform(), inplace=True)
    return out
