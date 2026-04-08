import geopandas as gpd
import xarray as xr

from .inputs import identify_network_pixels, align_inputs
from .centerlines import build_cost_graph, route_segments
from .skeleton import (
    raster_skeletonization,
    raster_skeleton_to_graph,
    filter_inlets,
    mask_constrained_snap,
    route_paths,
)
from .conversions import segments_to_gdf, segments_to_raster
from .widths import compute_widths


def derive_axis(
    dem: xr.DataArray,
    region_raster: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    method: str = "mcp",
    **kwargs,
) -> tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray]:
    """Derives valley centerlines and computes continuous valley widths.

    Args:
        dem (xr.DataArray): Digital elevation model array with spatial metadata.
        region_raster (xr.DataArray): Valley floor mask where valid areas equal 1.
        flowlines (gpd.GeoDataFrame): Input stream network linestrings.
        method (str): 'mcp' for Kienholz cost-graph routing, 'skeleton' for medial axis routing.
        **kwargs: Additional parameters passed to the routing methods.

    Returns:
        Tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray]:
            - centerlines_gdf: Vector linestrings of the routed centerlines.
            - centerlines_raster: Raster representation of the centerlines.
            - widths_raster: Continuous valley width raster interpolated via IDW.
    """
    dem, region_raster, flowlines = align_inputs(dem, region_raster, flowlines)
    networks_dict = identify_network_pixels(flowlines, region_raster)
    pixel_size = dem.rio.resolution()[0]

    if method == "mcp":
        all_segments = _glacier_centerline(
            dem.values, region_raster.values, networks_dict, pixel_size, **kwargs
        )
    elif method == "skeleton":
        all_segments = _raster_skeleton(
            region_raster.values, networks_dict, pixel_size, **kwargs
        )
    else:
        raise ValueError("Method must be 'mcp' or 'skeleton'")

    centerlines_gdf = segments_to_gdf(all_segments, dem.rio.transform(), dem.rio.crs)
    centerlines_raster = segments_to_raster(all_segments, region_raster)

    widths = compute_widths(centerlines_raster.values, region_raster.values, pixel_size)
    widths = xr.DataArray(
        widths,
        coords=region_raster.coords,
        dims=region_raster.dims,
    )
    widths = widths.rio.write_crs(region_raster.rio.crs)

    return centerlines_gdf, centerlines_raster, widths


def _glacier_centerline(dem_array, region_array, networks_dict, pixel_size, **kwargs):
    """Executes the Kienholz topographic cost-graph methodology."""
    mcp = build_cost_graph(dem_array, region_array, pixel_size, **kwargs)
    all_segments = {}

    for net_id, (inlets, outlet) in networks_dict.items():
        segments = route_segments(mcp, inlets, outlet)
        all_segments[net_id] = segments

    return all_segments


def _raster_skeleton(region_array, networks_dict, pixel_size, **kwargs):
    """Executes the morphological medial axis (skeleton) methodology."""
    # Note: skeleton doesn't need DEM data, only the binary valley mask
    skeleton_array = raster_skeletonization(region_array, **kwargs)
    graph = raster_skeleton_to_graph(skeleton_array)

    all_segments = {}

    for net_id, (inlets, outlet) in networks_dict.items():
        filtered_inlets = filter_inlets(inlets, region_array, pixel_size, **kwargs)
        snapped_inlets, snapped_outlet = mask_constrained_snap(
            filtered_inlets, outlet, skeleton_array, region_array
        )
        segments = route_paths(graph, snapped_inlets, snapped_outlet)
        all_segments[net_id] = segments

    return all_segments
