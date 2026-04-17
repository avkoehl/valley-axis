import numpy as np
import numpy.ma as ma
import xarray as xr
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import skfmm

from .centerlines import Centerlines


def get_allocation(
    centerlines: Centerlines,
    mask: xr.DataArray,
) -> xr.DataArray:
    """
    Partition the valley mask into one region per centerline segment.

    Each pixel is assigned the segment_id of the centerline whose geodesic
    territory it falls within. Claim order is hierarchical: segments of
    lower path_label (main stem) claim before tributaries. Pixels outside
    the mask are 0.

    Returns a raster labeled by segment_id.
    """
    segment_raster = centerlines.label_by_segment()
    segment_array = segment_raster.values
    mask_array = mask.values == 1

    allocation = np.zeros(mask_array.shape, dtype=np.uint32)
    claimed = np.zeros_like(mask_array, dtype=bool)

    radius_map = distance_transform_edt(mask_array)

    # Claim order: (path_label asc, segment_id asc) — main stem first.
    claim_order = centerlines.segments.sort_values(["path_label", "segment_id"])[
        "segment_id"
    ].tolist()

    for seg_id in claim_order:
        seg_mask = segment_array == seg_id
        if not np.any(seg_mask):
            continue

        phi = np.ones_like(mask_array, dtype=float)
        phi[seg_mask] = 0.0
        obstacles = (~mask_array) | claimed
        phi_masked = ma.MaskedArray(phi, mask=obstacles)

        try:
            dist, extended_radii = skfmm.extension_velocities(phi_masked, radius_map)
        except ValueError:
            continue

        valid_claim = (
            mask_array & ~claimed & ~dist.mask & (dist.data <= extended_radii.data)
        )
        allocation[valid_claim] = int(seg_id)
        claimed[valid_claim] = True

    # Orphan resolution: Geodesic categorical expansion
    unclaimed = mask_array & ~claimed
    if np.any(unclaimed) and np.any(claimed):
        # Create a topography where claimed pixels are valleys (0)
        # and unclaimed pixels are hills.
        dist_from_claimed = distance_transform_edt(~claimed)

        # Watershed safely expands the discrete integers in 'allocation'
        # outward into the 0-valued areas, constrained entirely by 'mask_array'.
        allocation = watershed(
            image=dist_from_claimed, markers=allocation, mask=mask_array
        ).astype(np.uint32)

    out = xr.DataArray(allocation, coords=mask.coords, dims=mask.dims, attrs=mask.attrs)
    out.rio.write_crs(mask.rio.crs, inplace=True)
    out.rio.write_transform(mask.rio.transform(), inplace=True)
    return out
