import geopandas as gpd
import networkx as nx
import rasterio
from shapely.geometry import Point
import xarray as xr

from .glacial import build_cost_graph, route_segments
from .skeleton import (
    raster_skeletonization,
    raster_skeleton_to_graph,
    filter_inlets,
    mask_constrained_snap,
    route_paths,
)
from .hierarchy import calculate_hierarchy
from .conversions import (
    lines_to_network,
    build_gdf,
    build_raster,
)
from ..inputs import align_inputs


def get_centerlines(
    dem: xr.DataArray,
    region_raster: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    method: str = "mcp",
    **kwargs,
) -> tuple[gpd.GeoDataFrame, xr.DataArray]:
    """Derives valley centerlines from a valley floor mask.

    Args:
        dem: Digital elevation model.
        region_raster: Binary valley floor mask (1 = valid).
        flowlines: Stream network linestrings.
        method: 'mcp' for Kienholz cost-graph routing, 'skeleton' for medial axis.
        **kwargs: Passed to the underlying routing method.

    Returns:
        centerlines_gdf: Vector linestrings of the routed centerlines with hierarchy.
        centerlines_raster: Raster representation of the centerlines (Strahler order).
    """
    pixel_size = float(abs(dem.rio.resolution()[0]))
    dem, region_raster, flowlines = align_inputs(dem, region_raster, flowlines)
    networks_dict = _identify_network_pixels(flowlines, region_raster)

    # 1. Route to get raw pixel segments
    if method == "mcp":
        raw_segments_dict = _run_glacial(
            dem.values, region_raster.values, networks_dict, pixel_size, **kwargs
        )
    elif method == "skeleton":
        raw_segments_dict = _run_skeleton(
            region_raster.values, networks_dict, pixel_size, **kwargs
        )
    else:
        raise ValueError(f"method must be 'mcp' or 'skeleton', got '{method}'")

    # 2. Calculate hierarchy (Strahler & Main Stems) for each network
    labeled_segments = {}
    for net_id, raw_segs in raw_segments_dict.items():
        labeled_segments[net_id] = calculate_hierarchy(raw_segs)

    # 3. Build final outputs
    centerlines_gdf = build_gdf(labeled_segments, dem.rio.transform(), dem.rio.crs)
    centerlines_raster = build_raster(labeled_segments, region_raster)

    return centerlines_gdf, centerlines_raster


# --- Core Routing Functions ---


def _run_glacial(dem_array, region_array, networks_dict, pixel_size, **kwargs):
    mcp = build_cost_graph(dem_array, region_array, pixel_size, **kwargs)
    all_segments = {}
    for net_id, (inlets, outlet) in networks_dict.items():
        all_segments[net_id] = route_segments(mcp, inlets, outlet)
    return all_segments


def _run_skeleton(region_array, networks_dict, pixel_size, **kwargs):
    skeleton_array = raster_skeletonization(region_array, **kwargs)
    graph = raster_skeleton_to_graph(skeleton_array)
    all_segments = {}
    for net_id, (inlets, outlet) in networks_dict.items():
        filtered_inlets = filter_inlets(inlets, region_array, pixel_size, **kwargs)
        snapped_inlets, snapped_outlet = mask_constrained_snap(
            filtered_inlets, outlet, skeleton_array, region_array
        )
        all_segments[net_id] = route_paths(graph, snapped_inlets, snapped_outlet)
    return all_segments


# --- Network Identification Functions ---


def _identify_network_pixels(
    flowlines_gdf: gpd.GeoDataFrame, region_raster: xr.DataArray
) -> dict[int, tuple[list[tuple[int, int]], tuple[int, int]]]:
    """Extracts valid inlet and outlet pixel indices for each distinct river network."""
    flowlines = _assign_network_ids(flowlines_gdf.copy())
    transform = region_raster.rio.transform()
    mask_array = region_raster.values
    networks_dict = {}

    for net_id, group in flowlines.groupby("network_id"):
        inflow_points, outlet_point = _find_network_endpoints(group)

        if not outlet_point:
            continue

        outlet_pixel = _point_to_valid_pixel(outlet_point, transform, mask_array)
        if not outlet_pixel:
            continue

        inlet_pixels = []
        for pt in inflow_points:
            pixel = _point_to_valid_pixel(pt, transform, mask_array)
            if pixel:
                inlet_pixels.append(pixel)

        if inlet_pixels and outlet_pixel:
            networks_dict[net_id] = (inlet_pixels, outlet_pixel)

    return networks_dict


def _assign_network_ids(flowlines):
    """Identifies distinct river networks and assigns a unique network_id."""
    graph = lines_to_network(flowlines)
    outlets = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    if len(outlets) == 1:
        flowlines["network_id"] = 1
        return flowlines

    flowlines["network_id"] = None
    for i, outlet in enumerate(outlets, start=1):
        upstream = nx.ancestors(graph, outlet)
        upstream.add(outlet)
        subgraph = graph.subgraph(upstream)
        streams = list(
            set(data["streamID"] for _, _, data in subgraph.edges(data=True))
        )
        flowlines.loc[streams, "network_id"] = i

    return flowlines


def _find_network_endpoints(single_network_gdf):
    """Finds inflow points and the outlet point for a SINGLE network."""
    startpoints = set()
    endpoints = set()

    for line in single_network_gdf.geometry:
        coords = list(line.coords)
        startpoints.add(tuple(coords[0]))
        endpoints.add(tuple(coords[-1]))

    inflows = [Point(pt) for pt in startpoints - endpoints]
    outlets = [Point(pt) for pt in endpoints - startpoints]
    outlet = outlets[0] if outlets else None

    return inflows, outlet


def _point_to_valid_pixel(point, transform, mask_array, search_radius=3):
    """Snaps a spatial point to the nearest valid pixel within a search radius."""
    r, c = rasterio.transform.rowcol(transform, point.x, point.y)
    height, width = mask_array.shape

    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask_array[nr, nc] == 1:
                return (nr, nc)
    return None
