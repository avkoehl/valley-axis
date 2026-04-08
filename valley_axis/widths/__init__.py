import numpy as np
import xarray as xr

from .idw import compute_idw
from .voronoi import compute_voronoi
from .laplace import compute_laplace

_METHODS = {
    "idw": compute_idw,
    "voronoi": compute_voronoi,
    "laplace": compute_laplace,
}


def get_widths(
    centerlines_raster: xr.DataArray,
    region_raster: xr.DataArray,
    method: str = "laplace",
    **kwargs,
) -> xr.DataArray:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}, got '{method}'")

    pixel_size = float(abs(region_raster.rio.resolution()[0]))
    mask = region_raster.values == 1

    result = _METHODS[method](centerlines_raster.values, mask, pixel_size, **kwargs)

    widths = xr.DataArray(result, coords=region_raster.coords, dims=region_raster.dims)
    widths = widths.where(mask)
    widths = widths.rio.write_crs(region_raster.rio.crs)
    widths = widths.rio.write_nodata(np.nan)

    return widths
