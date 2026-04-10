import numpy as np
import xarray as xr
import geopandas as gpd

from .idw import compute_idw
from .voronoi import compute_voronoi
from .laplace import compute_laplace
from .hierarchical import compute_hierarchical_voronoi

_METHODS = {
    "idw": compute_idw,
    "voronoi": compute_voronoi,
    "laplace": compute_laplace,
    "hierarchical": compute_hierarchical_voronoi,
}


def get_widths(
    centerlines_raster: xr.DataArray,
    region_raster: xr.DataArray,
    method: str = "laplace",
    centerlines_gdf: gpd.GeoDataFrame | None = None,
    **kwargs,
) -> xr.DataArray:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {list(_METHODS)}, got '{method}'")

    pixel_size = float(abs(region_raster.rio.resolution()[0]))
    mask = region_raster.values == 1

    # -- Dispatch Logic --
    if method == "hierarchical":
        if centerlines_gdf is None:
            raise ValueError(
                "The 'hierarchical' method requires 'centerlines_gdf' to be provided."
            )
        result, _ = _METHODS[method](
            centerline_array=centerlines_raster.values,
            mask_array=mask,
            pixel_size=pixel_size,
            centerlines_gdf=centerlines_gdf,
        )
    else:
        # Standard methods
        result = _METHODS[method](centerlines_raster.values, mask, pixel_size, **kwargs)

    # -- Format Output --
    widths = xr.DataArray(result, coords=region_raster.coords, dims=region_raster.dims)
    widths = widths.where(mask)
    widths = widths.rio.write_crs(region_raster.rio.crs)
    widths = widths.rio.write_nodata(np.nan)

    return widths
