import geopandas as gpd
import networkx as nx
import rasterio
from rasterio.features import shapes
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union
import xarray as xr

from .conversions import lines_to_network


def align_inputs(dem, region, flowlines):
    dem = dem.rio.reproject_match(region)
    flowlines = flowlines.to_crs(dem.rio.crs)
    valid_mask = (dem != dem.rio.nodata) & (region != region.rio.nodata)
    dem = dem.where(valid_mask, dem.rio.nodata)
    region = region.where(valid_mask, region.rio.nodata)

    polygons = [
        shape(geom)
        for geom, val in shapes(
            valid_mask.values.astype("uint8"),
            transform=region.rio.transform(),
            connectivity=8,
        )
        if val == 1
    ]
    valid_geom = unary_union(polygons)
    flowlines = (
        flowlines.clip(valid_geom).explode(index_parts=False).reset_index(drop=True)
    )
    return dem, region, flowlines


def identify_network_pixels(
    flowlines_gdf: gpd.GeoDataFrame, region_raster: xr.DataArray
) -> dict[int, tuple[list[tuple[int, int]], tuple[int, int]]]:
    """Extracts valid inlet and outlet pixel indices for each distinct river network.

    This function identifies separate river networks within the provided flowlines,
    finds their topological channel heads (inlets) and outlets, and converts those
    spatial coordinates into array indices (row, column) based on the provided
    region raster. Points falling outside the region mask are discarded.

    Args:
        flowlines_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the stream
            network linestrings.
        region_raster (xr.DataArray): A raster array representing the valley region
            mask. Must possess a `.rio.transform()` method and a `.values` attribute,
            where valid areas equal 1.

    Returns:
        Dict[int, Tuple[List[Tuple[int, int]], Tuple[int, int]]]: A dictionary mapping
        each `network_id` to its routing endpoints. The value is a tuple containing:
            - A list of inlet pixel coordinates: `[(row1, col1), (row2, col2), ...]`
            - A single outlet pixel coordinate: `(row, col)`
    """
    flowlines = _assign_network_ids(flowlines_gdf.copy())

    transform = region_raster.rio.transform()
    mask_array = region_raster.values

    networks_dict = {}

    # Process one cleanly separated network at a time
    for net_id, group in flowlines.groupby("network_id"):
        inflow_points, outlet_point = _find_network_endpoints(group)

        if not outlet_point:
            continue

        # Validate the outlet
        outlet_pixel = _point_to_valid_pixel(outlet_point, transform, mask_array)
        if not outlet_pixel:
            continue

        # Validate the inlets
        inlet_pixels = []
        for pt in inflow_points:
            pixel = _point_to_valid_pixel(pt, transform, mask_array)
            if pixel:
                inlet_pixels.append(pixel)

        # Only keep networks with at least one valid inlet and an outlet
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
    """
    Finds inflow points (channel heads) and the outlet point for a SINGLE network.
    Returns a tuple: (list of inflow Points, outlet Point).
    """
    startpoints = set()
    endpoints = set()

    for line in single_network_gdf.geometry:
        coords = list(line.coords)
        startpoints.add(tuple(coords[0]))
        endpoints.add(tuple(coords[-1]))

    # Inflows: start points that are never an end point
    inflows = [Point(pt) for pt in startpoints - endpoints]

    # Outlets: end points that are never a start point
    outlets = [Point(pt) for pt in endpoints - startpoints]

    # Safely grab the single outlet (assuming clean topology)
    outlet = outlets[0] if outlets else None

    return inflows, outlet


def _point_to_valid_pixel(point, transform, mask_array, search_radius=3):
    r, c = rasterio.transform.rowcol(transform, point.x, point.y)
    height, width = mask_array.shape

    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask_array[nr, nc] == 1:
                return (nr, nc)
    return None
