from collections import defaultdict

import xarray as xr
import geopandas as gpd

from .centerlines import get_centerlines
from .widths import get_widths


def measure_width(
    dem: xr.DataArray,
    region_raster: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    centerline_method: str = "skeleton",
    width_method: str = "laplace",
    segmentation: bool = True,
    centerline_kwargs: dict | None = None,
    width_kwargs: dict | None = None,
) -> tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray, xr.DataArray]:
    centerline_kwargs = centerline_kwargs or {}
    width_kwargs = width_kwargs or {}

    centerlines_gdf, centerlines_raster, path_map = get_centerlines(
        dem, region_raster, flowlines, method=centerline_method, **centerline_kwargs
    )

    path_to_segments = defaultdict(list)
    for seg_id, path_id in zip(
        centerlines_gdf["segment_id"], centerlines_gdf["path_label"]
    ):
        path_to_segments[path_id].append(seg_id)

    widths = get_widths(
        centerlines_raster,
        region_raster,
        method=width_method,
        path_map=path_map if segmentation else None,
        path_to_segments=path_to_segments if segmentation else None,
        **width_kwargs,
    )

    return centerlines_gdf, centerlines_raster, path_map, widths


__all__ = ["measure_width", "get_centerlines", "get_widths"]
