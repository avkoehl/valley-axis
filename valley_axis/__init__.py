import xarray as xr
import geopandas as gpd

from .centerlines import get_centerlines
from .widths import get_widths


def measure_width(
    dem: xr.DataArray,
    region_raster: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    centerline_method: str = "mcp",
    width_method: str = "laplace",
    centerline_kwargs: dict | None = None,
    width_kwargs: dict | None = None,
) -> tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray]:
    centerline_kwargs = centerline_kwargs or {}
    width_kwargs = width_kwargs or {}

    centerlines_gdf, centerlines_raster = get_centerlines(
        dem, region_raster, flowlines, method=centerline_method, **centerline_kwargs
    )

    widths = get_widths(
        centerlines_raster,
        region_raster,
        method=width_method,
        centerlines_gdf=centerlines_gdf,
        **width_kwargs,
    )

    return centerlines_gdf, centerlines_raster, widths


__all__ = ["measure_width", "get_centerlines", "get_widths"]
