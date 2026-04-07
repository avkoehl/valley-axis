import rioxarray as rxr
import geopandas as gpd
import xarray as xr

from .inputs import identify_network_pixels
from .centerlines import build_cost_graph, route_segments
from .conversions import segments_to_gdf, segments_to_raster
from .widths import compute_widths


def derive_axis(
    dem: xr.DataArray,
    region_raster: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    **cost_kwargs,
) -> tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray]:
    """Derives valley centerlines and computes continuous valley widths.

    Args:
        dem (xr.DataArray): Digital elevation model array with spatial metadata.
        region_raster (xr.DataArray): Valley floor mask where valid areas equal 1.
        flowlines (gpd.GeoDataFrame): Input stream network linestrings.
        **cost_kwargs: Additional parameters passed to the cost graph builder
            (e.g., f1, a, f2, b).

    Returns:
        Tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray]:
            - centerlines_gdf: Vector linestrings of the routed centerlines.
            - centerlines_raster: Raster representation of the centerlines.
            - widths_raster: Continuous valley width raster interpolated via IDW.
    """
    dem = dem.rio.reproject_match(region_raster)
    networks_dict = identify_network_pixels(flowlines, region_raster)

    pixel_size = dem.rio.resolution()[0]
    mcp = build_cost_graph(dem.values, region_raster.values, pixel_size, **cost_kwargs)
    all_segments = {}
    for net_id, (inlets, outlet) in networks_dict.items():
        segments = route_segments(mcp, inlets, outlet)
        all_segments[net_id] = segments

    centerlines_gdf = segments_to_gdf(all_segments, dem.rio.transform(), dem.rio.crs)
    centerlines_raster = segments_to_raster(all_segments, region_raster)

    widths = compute_widths(centerlines_raster.values, region_raster.values, pixel_size)
    widths = xr.DataArray(
        widths,
        coords=region_raster.coords,
        dims=region_raster.dims,
        name="valley_width",
    )

    return centerlines_gdf, centerlines_raster, widths
