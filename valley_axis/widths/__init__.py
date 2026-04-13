import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

from .interpolation import compute_voronoi, compute_idw, compute_laplace

_METHODS = {
    "voronoi": compute_voronoi,
    "idw": compute_idw,
    "laplace": compute_laplace,
}


def get_widths(
    centerlines_raster: xr.DataArray,
    region_raster: xr.DataArray,
    method: str = "laplace",
    path_map: xr.DataArray | None = None,
    path_to_segments: dict | None = None,
    **kwargs,
) -> xr.DataArray:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}, got '{method}'")

    fn = _METHODS[method]
    pixel_size = float(abs(region_raster.rio.resolution()[0]))
    mask_array = region_raster.values == 1
    centerline_array = centerlines_raster.values

    # Compute exact widths once from the full valley mask
    radius_pixels = distance_transform_edt(mask_array)
    centerline_widths = np.where(
        centerline_array > 0, radius_pixels * pixel_size * 2, 0.0
    )

    out = np.full(mask_array.shape, np.nan, dtype=np.float64)

    if path_map is not None:
        path_ids = np.unique(path_map.values[mask_array])
        for path_id in path_ids:
            region_mask = (path_map.values == path_id) & mask_array

            if path_to_segments is None:
                raise ValueError(
                    "segment_to_path must be provided when path_map is used"
                )
            seg_ids = path_to_segments[path_id]
            region_centerlines = np.where(
                np.isin(centerline_array, seg_ids), centerline_array, 0
            )

            if not np.any(region_centerlines > 0):
                continue

            result = fn(region_centerlines, region_mask, centerline_widths, **kwargs)
            out[region_mask] = result[region_mask]
    else:
        out = fn(centerline_array, mask_array, centerline_widths, **kwargs)

    widths = xr.DataArray(out, coords=region_raster.coords, dims=region_raster.dims)
    widths = widths.where(mask_array)
    widths = widths.rio.write_crs(region_raster.rio.crs)
    widths = widths.rio.write_nodata(np.nan)

    return widths
